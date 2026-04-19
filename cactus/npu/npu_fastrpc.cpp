#include "npu_fastrpc.h"
#include "fastrpc_drv.h"
#include "htp_protocol.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <memory>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace cactus {
namespace npu {

// ---------------------------------------------------------------------------
// CACT weight file reader (shared with npu_qnn_direct.cpp)
// ---------------------------------------------------------------------------
static constexpr uint32_t CACT_MAGIC      = 0x54434143;
static constexpr size_t   CACT_HEADER_SIZE = 84;

struct CactWeight {
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE;
    HANDLE hm = nullptr;
#else
    int fd = -1;
#endif
    void*  base      = nullptr;
    size_t file_size = 0;

    uint32_t precision    = 0; // 0=INT8 1=FP16 2=FP32 3=INT4
    bool     is_interleaved = false;
    uint32_t group_size   = 0;
    uint32_t num_groups   = 0;
    uint64_t original_N   = 0;
    uint64_t byte_size    = 0;
    uint64_t scales_bytes = 0;
    size_t   data_offset  = 0;
    size_t   scales_offset = 0;
    std::vector<uint32_t> shape;

    ~CactWeight() {
#ifdef _WIN32
        if (base) UnmapViewOfFile(base);
        if (hm)   CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
#else
        if (base) munmap(base, file_size);
        if (fd >= 0) close(fd);
#endif
    }

    const void*   data_ptr()   const { return (const char*)base + data_offset; }
    const __fp16* scales_ptr() const {
        if (!scales_bytes) return nullptr;
        return (const __fp16*)((const char*)base + scales_offset);
    }
};

static inline size_t align_up(size_t v, size_t a) {
    size_t r = v % a; return r ? v + a - r : v;
}

static bool open_cact(const std::string& path, CactWeight& f) {
#ifdef _WIN32
    f.hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                       nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f.hf == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "[FastRPC] cannot open: %s\n", path.c_str());
        return false;
    }
    LARGE_INTEGER sz; GetFileSizeEx(f.hf, &sz); f.file_size = (size_t)sz.QuadPart;
    f.hm = CreateFileMappingA(f.hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!f.hm) return false;
    f.base = MapViewOfFile(f.hm, FILE_MAP_READ, 0, 0, 0);
    if (!f.base) return false;
#else
    f.fd = open(path.c_str(), O_RDONLY);
    if (f.fd < 0) { fprintf(stderr, "[FastRPC] cannot open: %s\n", path.c_str()); return false; }
    struct stat st; fstat(f.fd, &st); f.file_size = st.st_size;
    f.base = mmap(nullptr, f.file_size, PROT_READ, MAP_PRIVATE, f.fd, 0);
    if (f.base == MAP_FAILED) { f.base = nullptr; return false; }
#endif

    if (f.file_size < CACT_HEADER_SIZE) return false;
    const char* p = (const char*)f.base;
    size_t off = 0;

    uint32_t magic = *(const uint32_t*)(p + off); off += 4;
    if (magic != CACT_MAGIC) { fprintf(stderr, "[FastRPC] bad magic: %s\n", path.c_str()); return false; }

    uint32_t flags = *(const uint32_t*)(p + off); off += 4;
    f.is_interleaved = (flags & 8) != 0;
    uint32_t alignment = *(const uint32_t*)(p + off); off += 4;
    if (!alignment) alignment = 1;
    uint32_t ndim = *(const uint32_t*)(p + off); off += 4;

    f.shape.clear();
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t d = *(const uint64_t*)(p + off); off += 8;
        if (i < ndim && d > 0) f.shape.push_back((uint32_t)d);
    }

    f.precision    = *(const uint32_t*)(p + off); off += 4;
    f.byte_size    = *(const uint64_t*)(p + off); off += 8;
    f.scales_bytes = *(const uint64_t*)(p + off); off += 8;
    f.group_size   = *(const uint32_t*)(p + off); off += 4;
    f.num_groups   = *(const uint32_t*)(p + off); off += 4;
    f.original_N   = *(const uint64_t*)(p + off); off += 8;

    size_t ah = align_up(CACT_HEADER_SIZE, alignment);
    if (f.scales_bytes > 0) {
        f.scales_offset = ah;
        f.data_offset   = align_up(ah + f.scales_bytes, alignment);
    } else {
        f.data_offset   = ah;
    }
    return true;
}

static uint16_t f32_to_f16(float v) {
    if (v == 0.0f) return 0;
    uint32_t b; memcpy(&b, &v, 4);
    int e = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = (b >> 13) & 0x3FF, s = (b >> 31) << 15;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

// Dequantize any CACT precision to packed FP16 (row-major).
static std::vector<uint16_t> cact_to_fp16(const CactWeight& f) {
    if (f.precision == 1) {
        size_t n = 1; for (auto d : f.shape) n *= d;
        std::vector<uint16_t> out(n);
        memcpy(out.data(), f.data_ptr(), n * 2);
        return out;
    }
    if (f.precision == 3) {
        // INT4 interleaved (matches cactus CPU GEMV kernel layout)
        const uint8_t* packed = (const uint8_t*)f.data_ptr();
        const __fp16*  scales = f.scales_ptr();
        int gs = (int)f.group_size;
        size_t N = (f.is_interleaved && f.original_N > 0) ? (size_t)f.original_N : (size_t)f.shape[0];
        size_t K = f.shape.size() >= 2 ? (size_t)f.shape[1] : 1;
        int ng = gs > 0 ? (int)(K / gs) : 1;
        std::vector<uint16_t> out(N * K);

        auto nibble = [&](size_t bi, bool hi) -> int8_t {
            uint8_t b = packed[bi];
            return hi ? (int8_t)(b) >> 4 : (int8_t)((b & 0xF) << 4) >> 4;
        };

        if (f.is_interleaved) {
            for (size_t n = 0; n < N; n++) {
                int nb = (int)(n / 4), ni = (int)(n % 4);
                for (size_t k = 0; k < K; k++) {
                    float sc = 1.0f;
                    if (scales && gs > 0) {
                        int g = (int)(k / gs);
                        sc = (float)scales[(size_t)(nb * ng + g) * 4 + ni];
                    }
                    size_t kg = k / gs, kw = k % gs;
                    size_t kc = kw / 8, ki = kw % 4;
                    bool   hi = (kw % 8) >= 4;
                    size_t boff = ((size_t)nb * K + kg * (size_t)gs) * 2 + kc * 16 + (size_t)ni * 4 + ki;
                    out[n * K + k] = f32_to_f16((float)nibble(boff, hi) * sc);
                }
            }
        } else {
            size_t ne = N * K;
            for (size_t i = 0; i < ne; i++) {
                float sc = (scales && gs > 0) ? (float)scales[i / gs] : 1.0f;
                out[i] = f32_to_f16((float)nibble(i / 2, i & 1) * sc);
            }
        }
        return out;
    }
    if (f.precision == 0) {
        // INT8
        const int8_t* packed = (const int8_t*)f.data_ptr();
        const __fp16* scales = f.scales_ptr();
        int gs = (int)f.group_size;
        size_t N = f.shape[0];
        size_t K = f.shape.size() >= 2 ? (size_t)f.shape[1] : 1;
        std::vector<uint16_t> out(N * K);
        for (size_t i = 0; i < N * K; i++) {
            float sc = (scales && gs > 0) ? (float)scales[i / gs] : 1.0f;
            out[i] = f32_to_f16((float)packed[i] * sc);
        }
        return out;
    }
    fprintf(stderr, "[FastRPC] unsupported precision %u\n", f.precision);
    size_t n = 1; for (auto d : f.shape) n *= d;
    return std::vector<uint16_t>(n, 0);
}

// ---------------------------------------------------------------------------
// RpcMem RAII wrapper
// ---------------------------------------------------------------------------
struct RpcBuf {
    void*  base = nullptr;
    size_t size = 0;
    int    fd   = -1;

    bool alloc(size_t bytes) {
        // Round up to 4K
        size = (bytes + 4095) & ~(size_t)4095;
        base = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)size);
        if (!base) { fprintf(stderr, "[FastRPC] rpcmem_alloc(%zu) failed\n", size); return false; }
        fd = rpcmem_to_fd(base);
        if (fd < 0) { fprintf(stderr, "[FastRPC] rpcmem_to_fd failed\n"); return false; }
        return true;
    }

    ~RpcBuf() { if (base) rpcmem_free(base); }

    // No copy
    RpcBuf() = default;
    RpcBuf(const RpcBuf&) = delete;
    RpcBuf& operator=(const RpcBuf&) = delete;
};

// ---------------------------------------------------------------------------
// OpBatch builder
// ---------------------------------------------------------------------------
struct OpBatch {
    std::vector<htp_buf_desc>  bufs;
    std::vector<htp_tensor>    tensors;
    std::vector<htp_op_desc>   ops;

    // Scratch buffer (op_shm) — shared memory for the batch descriptor payload
    std::unique_ptr<RpcBuf> shm;

    bool init(size_t shm_bytes) {
        shm = std::make_unique<RpcBuf>();
        return shm->alloc(shm_bytes);
    }

    // Register a buffer and return its index
    uint16_t add_buf(const RpcBuf& buf) {
        htp_buf_desc d{};
        d.base  = (uint64_t)(uintptr_t)buf.base;
        d.size  = buf.size;
        d.flags = 0;
        d.fd    = (uint32_t)buf.fd;
        bufs.push_back(d);
        return (uint16_t)(bufs.size() - 1);
    }

    // Register a tensor and return its index
    uint16_t add_tensor(uint16_t buf_idx, uint32_t byte_offset, uint32_t byte_size,
                        htp_data_type dtype, uint32_t flags,
                        uint32_t ne0, uint32_t ne1 = 1, uint32_t ne2 = 1, uint32_t ne3 = 1) {
        htp_tensor t{};
        t.data  = byte_offset;
        t.size  = byte_size;
        t.flags = flags;
        t.type  = (uint16_t)dtype;
        t.bi    = buf_idx;
        t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
        // Row-major strides
        uint32_t esz = (dtype == HTP_TYPE_F16) ? 2 : (dtype == HTP_TYPE_F32) ? 4 : 2;
        t.nb[0] = esz;
        t.nb[1] = ne0 * esz;
        t.nb[2] = ne0 * ne1 * esz;
        t.nb[3] = ne0 * ne1 * ne2 * esz;
        tensors.push_back(t);
        return (uint16_t)(tensors.size() - 1);
    }

    // Add an op
    void add_op(htp_op_code opcode, uint16_t dst, std::initializer_list<uint16_t> srcs,
                const int32_t* params = nullptr, int n_params = 0) {
        htp_op_desc op{};
        op.opcode = (uint32_t)opcode;
        op.dst    = dst;
        int i = 0;
        for (auto s : srcs) { if (i < HTP_OP_MAX_INPUTS) op.src[i++] = s; }
        if (params) memcpy(op.params, params, n_params * sizeof(int32_t));
        ops.push_back(op);
    }

    // Serialize to shared memory and return size
    size_t flush(uint8_t* dst, size_t capacity) const {
        size_t sz_bufs    = bufs.size()    * sizeof(htp_buf_desc);
        size_t sz_tensors = tensors.size() * sizeof(htp_tensor);
        size_t sz_ops     = ops.size()     * sizeof(htp_op_desc);
        size_t total = sz_bufs + sz_tensors + sz_ops;
        if (total > capacity) {
            fprintf(stderr, "[FastRPC] batch too large: %zu > %zu\n", total, capacity);
            return 0;
        }
        uint8_t* p = dst;
        memcpy(p, bufs.data(),    sz_bufs);    p += sz_bufs;
        memcpy(p, tensors.data(), sz_tensors); p += sz_tensors;
        memcpy(p, ops.data(),     sz_ops);
        return total;
    }

    void clear() { bufs.clear(); tensors.clear(); ops.clear(); }
};

// ---------------------------------------------------------------------------
// Impl struct
// ---------------------------------------------------------------------------
struct FastRPCPrefill::Impl {
    // Model config
    int chunk_size    = 64;
    int hidden_dim    = 896;
    int num_layers    = 24;
    int num_kv_heads  = 2;
    int head_dim      = 64;
    int num_heads     = 14;
    int ffn_dim       = 4864;
    float layer_norm_eps = 1e-6f;
    float rope_theta     = 1000000.0f;

    bool loaded = false;
    int  arch   = 73;     // Hexagon DSP architecture version

    // FastRPC session
    remote_handle64 htp_handle  = 0;
    dspqueue_t      queue       = nullptr;
    uint64_t        queue_id    = 0;

    // Shared memory buffers:
    // One weight buffer per layer (all layer weights packed together).
    // One activation buffer (reused across layers, holds all intermediates).
    // One I/O buffer (embeddings in, K/V out).
    struct LayerWeights {
        std::unique_ptr<RpcBuf> buf;
        uint32_t off_in_norm;    // input_norm.weights      [hidden_dim]
        uint32_t off_post_norm;  // post_attn_norm.weights  [hidden_dim]
        uint32_t off_wq;         // attn_q.weights          [num_heads*head_dim, hidden_dim]
        uint32_t off_wk;         // attn_k.weights          [num_kv_heads*head_dim, hidden_dim]
        uint32_t off_wv;         // attn_v.weights          [num_kv_heads*head_dim, hidden_dim]
        uint32_t off_wo;         // attn_output.weights     [hidden_dim, num_heads*head_dim]
        uint32_t off_wgate;      // ffn_gate.weights        [ffn_dim, hidden_dim]
        uint32_t off_wup;        // ffn_up.weights          [ffn_dim, hidden_dim]
        uint32_t off_wdown;      // ffn_down.weights        [hidden_dim, ffn_dim]
    };
    std::vector<LayerWeights> layer_weights;

    std::unique_ptr<RpcBuf> act_buf;  // Scratch for activations (reused each layer)
    std::unique_ptr<RpcBuf> io_buf;   // Embeddings in + K/V cache outputs

    // Activation buffer layout offsets (all in act_buf)
    uint32_t act_off_normed;     // [chunk, hidden_dim] FP16
    uint32_t act_off_q;          // [chunk, num_heads, head_dim] FP16
    uint32_t act_off_k;          // [chunk, num_kv_heads, head_dim] FP16
    uint32_t act_off_v;          // [chunk, num_kv_heads, head_dim] FP16
    uint32_t act_off_attn;       // [chunk, hidden_dim] FP16
    uint32_t act_off_o;          // [chunk, hidden_dim] FP16
    uint32_t act_off_res1;       // [chunk, hidden_dim] FP16 (post-attention residual)
    uint32_t act_off_normed2;    // [chunk, hidden_dim] FP16
    uint32_t act_off_gate;       // [chunk, ffn_dim] FP16
    uint32_t act_off_up;         // [chunk, ffn_dim] FP16
    uint32_t act_off_swiglu;     // [chunk, ffn_dim] FP16
    uint32_t act_off_ffn_out;    // [chunk, hidden_dim] FP16
    uint32_t act_buf_size;

    // I/O buffer layout
    uint32_t io_off_emb;         // [chunk, hidden_dim] FP16  (input embeddings)
    uint32_t io_off_cos;         // [chunk, head_dim/2] FP16
    uint32_t io_off_sin;         // [chunk, head_dim/2] FP16
    // K/V outputs: [num_layers][chunk, num_kv_heads, head_dim]
    std::vector<uint32_t> io_off_k;
    std::vector<uint32_t> io_off_v;
    uint32_t io_buf_size;

    // K/V output staging (host-side copies to return via NPUPrefillDirectResult)
    std::vector<std::vector<uint16_t>> k_bufs; // [num_layers][chunk*kv_heads*head_dim]
    std::vector<std::vector<uint16_t>> v_bufs;

    // Scratch OpBatch (reused each layer)
    OpBatch batch;

    // ---- Methods ----
    bool parse_config(const std::string& folder);
    bool alloc_buffers();
    bool load_layer_weights(const std::string& folder, int li);
    bool open_session(int arch_ver);
    bool map_buffers();

    void build_rope_cos_sin(int position_offset);

    bool run_layer(int li, uint16_t bi_act, uint16_t bi_io, uint16_t& hidden_tidx);
    AEEResult flush_batch();

    void teardown();
};

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::parse_config(const std::string& folder) {
    std::string path = folder + "/config.txt";
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "[FastRPC] cannot open config.txt: %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if      (k == "hidden_dim")            hidden_dim   = std::stoi(v);
        else if (k == "num_layers")            num_layers   = std::stoi(v);
        else if (k == "attention_heads")       num_heads    = std::stoi(v);
        else if (k == "attention_kv_heads")    num_kv_heads = std::stoi(v);
        else if (k == "attention_head_dim")    head_dim     = std::stoi(v);
        else if (k == "ffn_intermediate_dim")  ffn_dim      = std::stoi(v);
        else if (k == "rope_theta")            rope_theta   = std::stof(v);
        else if (k == "layer_norm_eps")        layer_norm_eps = std::stof(v);
    }
    fprintf(stderr, "[FastRPC] config: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d ffn=%d\n",
            hidden_dim, num_layers, num_heads, num_kv_heads, head_dim, ffn_dim);
    return true;
}

// ---------------------------------------------------------------------------
// Buffer allocation
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::alloc_buffers() {
    int cs = chunk_size, hd = head_dim, hd2 = hd / 2;
    int nh = num_heads, nkv = num_kv_heads;
    int em = hidden_dim, ff = ffn_dim;

    // Activation buffer layout
    auto next = [](uint32_t& off, uint32_t bytes) -> uint32_t {
        uint32_t o = (off + 63) & ~63u; off = o + bytes; return o;
    };
    uint32_t off = 0;
    act_off_normed   = next(off, cs * em * 2);
    act_off_q        = next(off, cs * nh * hd * 2);
    act_off_k        = next(off, cs * nkv * hd * 2);
    act_off_v        = next(off, cs * nkv * hd * 2);
    act_off_attn     = next(off, cs * nh * hd * 2);  // flash attn output
    act_off_o        = next(off, cs * em * 2);
    act_off_res1     = next(off, cs * em * 2);
    act_off_normed2  = next(off, cs * em * 2);
    act_off_gate     = next(off, cs * ff * 2);
    act_off_up       = next(off, cs * ff * 2);
    act_off_swiglu   = next(off, cs * ff * 2);
    act_off_ffn_out  = next(off, cs * em * 2);
    act_buf_size = off;

    act_buf = std::make_unique<RpcBuf>();
    if (!act_buf->alloc(act_buf_size)) return false;

    // I/O buffer layout
    off = 0;
    io_off_emb = next(off, cs * em * 2);
    io_off_cos = next(off, cs * hd2 * 2);
    io_off_sin = next(off, cs * hd2 * 2);
    io_off_k.resize(num_layers);
    io_off_v.resize(num_layers);
    for (int li = 0; li < num_layers; li++) {
        io_off_k[li] = next(off, cs * nkv * hd * 2);
        io_off_v[li] = next(off, cs * nkv * hd * 2);
    }
    io_buf_size = off;

    io_buf = std::make_unique<RpcBuf>();
    if (!io_buf->alloc(io_buf_size)) return false;

    // K/V staging
    size_t kv_sz = (size_t)cs * nkv * hd;
    k_bufs.assign(num_layers, std::vector<uint16_t>(kv_sz));
    v_bufs.assign(num_layers, std::vector<uint16_t>(kv_sz));

    // Op batch scratch (enough for one layer's worth of ops + tensors + bufs)
    const size_t SHM_SIZE = 256 * 1024; // 256 KB
    if (!batch.init(SHM_SIZE)) return false;

    fprintf(stderr, "[FastRPC] activation buf %zu KB, I/O buf %zu KB\n",
            act_buf_size / 1024, io_buf_size / 1024);
    return true;
}

// ---------------------------------------------------------------------------
// Load one layer's weights
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::load_layer_weights(const std::string& folder, int li) {
    LayerWeights& lw = layer_weights[li];
    std::string pfx = folder + "/layer_" + std::to_string(li) + "_";

    // Compute total size needed
    struct WeightInfo { const char* suffix; uint32_t N, K; };
    int em = hidden_dim, nh = num_heads, nkv = num_kv_heads, hd = head_dim, ff = ffn_dim;
    WeightInfo infos[] = {
        { "input_norm.weights",    (uint32_t)em,           1           },
        { "post_attn_norm.weights",(uint32_t)em,           1           },
        { "attn_q.weights",        (uint32_t)(nh * hd),    (uint32_t)em },
        { "attn_k.weights",        (uint32_t)(nkv * hd),   (uint32_t)em },
        { "attn_v.weights",        (uint32_t)(nkv * hd),   (uint32_t)em },
        { "attn_output.weights",   (uint32_t)em,           (uint32_t)(nh * hd) },
        { "ffn_gate.weights",      (uint32_t)ff,           (uint32_t)em },
        { "ffn_up.weights",        (uint32_t)ff,           (uint32_t)em },
        { "ffn_down.weights",      (uint32_t)em,           (uint32_t)ff },
    };
    constexpr int NWEIGHTS = 9;

    // Load all weights to FP16 first, compute total buffer size
    std::vector<std::vector<uint16_t>> fp16_data(NWEIGHTS);
    size_t total_bytes = 0;
    for (int i = 0; i < NWEIGHTS; i++) {
        std::string path = pfx + infos[i].suffix;
        CactWeight cw;
        if (!open_cact(path, cw)) {
            fprintf(stderr, "[FastRPC] missing weight: %s\n", path.c_str());
            return false;
        }
        fp16_data[i] = cact_to_fp16(cw);
        total_bytes += (fp16_data[i].size() * 2 + 63) & ~63ull;
    }

    lw.buf = std::make_unique<RpcBuf>();
    if (!lw.buf->alloc(total_bytes)) return false;

    uint8_t* dst = (uint8_t*)lw.buf->base;
    uint32_t off = 0;
    uint32_t* offsets[] = {
        &lw.off_in_norm, &lw.off_post_norm,
        &lw.off_wq, &lw.off_wk, &lw.off_wv, &lw.off_wo,
        &lw.off_wgate, &lw.off_wup, &lw.off_wdown,
    };
    for (int i = 0; i < NWEIGHTS; i++) {
        *offsets[i] = off;
        size_t bytes = fp16_data[i].size() * 2;
        memcpy(dst + off, fp16_data[i].data(), bytes);
        off = (uint32_t)((off + bytes + 63) & ~63ull);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Open FastRPC session with libggml-htp-vNN.so
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::open_session(int arch_ver) {
    // Enable unsigned DSP module loading
    struct remote_rpc_control_unsigned_module u;
    u.domain = 3; // CDSP
    u.enable = 1;
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] failed to enable unsigned PD: 0x%x\n", err);
        // Non-fatal: may already be enabled or system may allow unsigned skels
    }

    // Build URI for the skel
    char uri[256];
    snprintf(uri, sizeof(uri),
             "file:///libggml-htp-v%d.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp",
             arch_ver);

    err = htp_iface_open(uri, &htp_handle);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] htp_iface_open failed: 0x%x  uri=%s\n", err, uri);
        return false;
    }
    fprintf(stderr, "[FastRPC] opened session with %s\n", uri);

    // Create dspqueue for async op dispatch
    err = dspqueue_create(3 /*CDSP*/, 0,
                          4 * 1024 * 1024, // req queue 4MB
                          64 * 1024,       // resp queue 64KB
                          nullptr, nullptr, nullptr,
                          &queue);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] dspqueue_create failed: 0x%x\n", err);
        return false;
    }

    err = dspqueue_export(queue, &queue_id);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] dspqueue_export failed: 0x%x\n", err);
        return false;
    }

    // Start the DSP session
    err = htp_iface_start(htp_handle, 0 /*sess_id*/, queue_id, 0 /*hvx*/, 0 /*hmx*/);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] htp_iface_start failed: 0x%x\n", err);
        return false;
    }

    fprintf(stderr, "[FastRPC] DSP session started, queue_id=%llu\n", (unsigned long long)queue_id);
    return true;
}

// ---------------------------------------------------------------------------
// Map all weight buffers into DSP address space
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::map_buffers() {
    auto map_one = [&](const RpcBuf& buf, bool pinned) -> bool {
        AEEResult err = htp_iface_mmap(htp_handle, (uint32_t)buf.fd,
                                       (uint32_t)buf.size, pinned ? 1 : 0);
        if (err != AEE_SUCCESS) {
            fprintf(stderr, "[FastRPC] htp_iface_mmap failed: 0x%x (fd=%d size=%zu)\n",
                    err, buf.fd, buf.size);
            return false;
        }
        return true;
    };

    for (int li = 0; li < num_layers; li++) {
        if (!map_one(*layer_weights[li].buf, true /*pinned weight*/)) return false;
    }
    if (!map_one(*act_buf, false)) return false;
    if (!map_one(*io_buf, false)) return false;
    if (!map_one(*batch.shm, false)) return false;

    fprintf(stderr, "[FastRPC] all buffers mapped into DSP\n");
    return true;
}

// ---------------------------------------------------------------------------
// Build cos/sin tables for RoPE at a given position offset
// ---------------------------------------------------------------------------
void FastRPCPrefill::Impl::build_rope_cos_sin(int position_offset) {
    int cs = chunk_size, hd2 = head_dim / 2;
    uint16_t* cos_dst = (uint16_t*)((uint8_t*)io_buf->base + io_off_cos);
    uint16_t* sin_dst = (uint16_t*)((uint8_t*)io_buf->base + io_off_sin);

    for (int t = 0; t < cs; t++) {
        int pos = position_offset + t;
        for (int i = 0; i < hd2; i++) {
            float theta = (float)pos / powf(rope_theta, 2.0f * (float)i / (float)head_dim);
            cos_dst[t * hd2 + i] = f32_to_f16(cosf(theta));
            sin_dst[t * hd2 + i] = f32_to_f16(sinf(theta));
        }
    }
}

// ---------------------------------------------------------------------------
// Flush the current batch to the DSP and wait for response
// ---------------------------------------------------------------------------
AEEResult FastRPCPrefill::Impl::flush_batch() {
    if (batch.ops.empty()) return AEE_SUCCESS;

    htp_opbatch_req req;
    req.n_bufs    = (uint32_t)batch.bufs.size();
    req.n_tensors = (uint32_t)batch.tensors.size();
    req.n_ops     = (uint32_t)batch.ops.size();
    req.flags     = 0;

    uint8_t* shm_ptr = (uint8_t*)batch.shm->base;
    size_t payload_size = batch.flush(shm_ptr, batch.shm->size);
    if (!payload_size) return (AEEResult)AEE_EUNABLETOLOAD;

    struct dspqueue_buffer dbuf{};
    dbuf.fd     = batch.shm->fd;
    dbuf.flags  = DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER | DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    dbuf.ptr    = shm_ptr;
    dbuf.offset = 0;
    dbuf.size   = (uint32_t)payload_size;

    AEEResult err = dspqueue_write(queue,
                                   0, 1, &dbuf,
                                   sizeof(req), (const uint8_t*)&req,
                                   DSPQUEUE_TIMEOUT);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] dspqueue_write failed: 0x%x\n", err);
        return err;
    }

    // Wait for response
    htp_opbatch_rsp rsp{};
    struct dspqueue_buffer rbuf{};
    uint32_t flags_out = 0, nb_out = 0, msg_len = 0;
    err = dspqueue_read(queue, &flags_out, 1, &nb_out, &rbuf,
                        sizeof(rsp), &msg_len, (uint8_t*)&rsp, DSPQUEUE_TIMEOUT);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] dspqueue_read failed: 0x%x\n", err);
        return err;
    }
    if (rsp.status != HTP_STATUS_OK) {
        fprintf(stderr, "[FastRPC] DSP op failed: status=%u\n", rsp.status);
        return (AEEResult)AEE_EUNABLETOLOAD;
    }

    batch.clear();
    return AEE_SUCCESS;
}

// ---------------------------------------------------------------------------
// Run one transformer layer via DSP
// Returns false on error. hidden_tidx is updated to the output tensor index.
// ---------------------------------------------------------------------------
bool FastRPCPrefill::Impl::run_layer(int li, uint16_t bi_act, uint16_t bi_io,
                                     uint16_t& hidden_tidx) {
    batch.clear();

    const LayerWeights& lw = layer_weights[li];

    // Register buffers for this layer
    uint16_t bi_wt  = batch.add_buf(*lw.buf);
    // bi_act and bi_io are already registered in the caller
    (void)bi_act; (void)bi_io;
    // Actually we re-register each batch since batch is cleared
    bi_act = batch.add_buf(*act_buf);
    bi_io  = batch.add_buf(*io_buf);
    bi_wt  = batch.add_buf(*lw.buf);

    int cs = chunk_size, hd = head_dim, hd2 = hd / 2;
    int nh = num_heads, nkv = num_kv_heads;
    int em = hidden_dim, ff = ffn_dim;

    // Helper to add a weight tensor
    auto wt = [&](uint32_t off, uint32_t N, uint32_t K) -> uint16_t {
        uint32_t bytes = (K == 1) ? N * 2 : N * K * 2;
        return batch.add_tensor(bi_wt, off, bytes, HTP_TYPE_F16, 0,
                                (K == 1) ? N : K, (K == 1) ? 1 : N);
    };
    // Helper to add a compute tensor
    auto ct = [&](uint32_t off, uint32_t ne0, uint32_t ne1 = 1) -> uint16_t {
        return batch.add_tensor(bi_act, off, ne0 * ne1 * 2, HTP_TYPE_F16,
                                HTP_TENSOR_COMPUTE, ne0, ne1);
    };
    // Helper to add an I/O tensor
    auto iot = [&](uint32_t off, uint32_t ne0, uint32_t ne1 = 1) -> uint16_t {
        return batch.add_tensor(bi_io, off, ne0 * ne1 * 2, HTP_TYPE_F16, 0, ne0, ne1);
    };

    // Hidden state: input comes from io_buf (embeddings for layer 0) or act_buf (residual for layer 1+)
    // For simplicity, we always use the io_buf embedding slot as the hidden state,
    // but for layers > 0 we copy the residual there before calling this function.
    uint16_t t_hidden = iot((uint32_t)io_off_emb, (uint32_t)cs, (uint32_t)em);

    // Norm weights
    uint16_t t_in_norm   = wt(lw.off_in_norm,   (uint32_t)em, 1);
    uint16_t t_post_norm = wt(lw.off_post_norm,  (uint32_t)em, 1);

    // Weight matrices
    uint16_t t_wq   = wt(lw.off_wq,   (uint32_t)(nh  * hd), (uint32_t)em);
    uint16_t t_wk   = wt(lw.off_wk,   (uint32_t)(nkv * hd), (uint32_t)em);
    uint16_t t_wv   = wt(lw.off_wv,   (uint32_t)(nkv * hd), (uint32_t)em);
    uint16_t t_wo   = wt(lw.off_wo,   (uint32_t)em,          (uint32_t)(nh  * hd));
    uint16_t t_wgate = wt(lw.off_wgate, (uint32_t)ff,        (uint32_t)em);
    uint16_t t_wup   = wt(lw.off_wup,   (uint32_t)ff,        (uint32_t)em);
    uint16_t t_wdown  = wt(lw.off_wdown, (uint32_t)em,       (uint32_t)ff);

    // Compute tensors
    uint16_t t_normed  = ct((uint32_t)act_off_normed,  (uint32_t)cs, (uint32_t)em);
    uint16_t t_q       = ct((uint32_t)act_off_q,        (uint32_t)cs, (uint32_t)(nh  * hd));
    uint16_t t_k       = ct((uint32_t)act_off_k,        (uint32_t)cs, (uint32_t)(nkv * hd));
    uint16_t t_v       = ct((uint32_t)act_off_v,        (uint32_t)cs, (uint32_t)(nkv * hd));
    uint16_t t_attn    = ct((uint32_t)act_off_attn,     (uint32_t)cs, (uint32_t)(nh  * hd));
    uint16_t t_oproj   = ct((uint32_t)act_off_o,        (uint32_t)cs, (uint32_t)em);
    uint16_t t_res1    = ct((uint32_t)act_off_res1,     (uint32_t)cs, (uint32_t)em);
    uint16_t t_normed2 = ct((uint32_t)act_off_normed2,  (uint32_t)cs, (uint32_t)em);
    uint16_t t_gate    = ct((uint32_t)act_off_gate,     (uint32_t)cs, (uint32_t)ff);
    uint16_t t_up      = ct((uint32_t)act_off_up,       (uint32_t)cs, (uint32_t)ff);
    uint16_t t_swiglu  = ct((uint32_t)act_off_swiglu,   (uint32_t)cs, (uint32_t)ff);
    uint16_t t_ffnout  = ct((uint32_t)act_off_ffn_out,  (uint32_t)cs, (uint32_t)em);

    // RoPE tables
    uint16_t t_cos = iot((uint32_t)io_off_cos, (uint32_t)cs, (uint32_t)hd2);
    uint16_t t_sin = iot((uint32_t)io_off_sin, (uint32_t)cs, (uint32_t)hd2);

    // K/V outputs (in io_buf so host can read them)
    uint16_t t_k_out = iot((uint32_t)io_off_k[li], (uint32_t)cs, (uint32_t)(nkv * hd));
    uint16_t t_v_out = iot((uint32_t)io_off_v[li], (uint32_t)cs, (uint32_t)(nkv * hd));

    // ---- Build op graph for this layer ----

    // 1. RMS norm: normed = rms_norm(hidden, in_norm_w)
    {
        int32_t params[HTP_OP_MAX_PARAMS] = {};
        float eps = layer_norm_eps;
        memcpy(&params[0], &eps, 4);
        batch.add_op(HTP_OP_RMS_NORM, t_normed, {t_hidden, t_in_norm}, params, 1);
    }

    // 2. Q projection: q = normed * wq^T
    batch.add_op(HTP_OP_MUL_MAT, t_q, {t_wq, t_normed});

    // 3. K projection: k = normed * wk^T
    batch.add_op(HTP_OP_MUL_MAT, t_k, {t_wk, t_normed});

    // 4. V projection: v = normed * wv^T
    batch.add_op(HTP_OP_MUL_MAT, t_v, {t_wv, t_normed});

    // 5. RoPE on Q
    {
        int32_t params[HTP_OP_MAX_PARAMS] = {};
        params[0] = HTP_ROPE_TYPE_NEOX;  // mode
        // params[1] = n_dims, params[2] = freq_base, etc.
        params[2] = 0; // n_ctx (unused by skel)
        float freq_base = rope_theta;
        memcpy(&params[3], &freq_base, 4);
        batch.add_op(HTP_OP_ROPE, t_q, {t_q, t_cos, t_sin}, params, 4);
    }

    // 6. RoPE on K
    {
        int32_t params[HTP_OP_MAX_PARAMS] = {};
        params[0] = HTP_ROPE_TYPE_NEOX;
        float freq_base = rope_theta;
        memcpy(&params[3], &freq_base, 4);
        batch.add_op(HTP_OP_ROPE, t_k, {t_k, t_cos, t_sin}, params, 4);
    }

    // 7. Flash attention: attn = flash_attn_ext(q, k, v)
    {
        int32_t params[HTP_OP_MAX_PARAMS] = {};
        float scale = 1.0f / sqrtf((float)hd);
        memcpy(&params[0], &scale, 4);
        float zero = 0.0f;
        memcpy(&params[1], &zero, 4); // max_bias
        memcpy(&params[2], &zero, 4); // logit_softcap
        batch.add_op(HTP_OP_FLASH_ATTN_EXT, t_attn, {t_q, t_k, t_v}, params, 4);
    }

    // 8. Output projection: oproj = attn * wo^T
    batch.add_op(HTP_OP_MUL_MAT, t_oproj, {t_wo, t_attn});

    // 9. Residual: res1 = hidden + oproj
    batch.add_op(HTP_OP_ADD, t_res1, {t_hidden, t_oproj});

    // 10. Post-attention RMS norm: normed2 = rms_norm(res1, post_norm_w)
    {
        int32_t params[HTP_OP_MAX_PARAMS] = {};
        float eps = layer_norm_eps;
        memcpy(&params[0], &eps, 4);
        batch.add_op(HTP_OP_RMS_NORM, t_normed2, {t_res1, t_post_norm}, params, 1);
    }

    // 11. Gate / Up projections
    batch.add_op(HTP_OP_MUL_MAT, t_gate, {t_wgate, t_normed2});
    batch.add_op(HTP_OP_MUL_MAT, t_up,   {t_wup,   t_normed2});

    // 12. SwiGLU: swiglu = silu(gate) * up
    batch.add_op(HTP_OP_GLU_SWIGLU, t_swiglu, {t_gate, t_up});

    // 13. Down projection: ffn_out = swiglu * wdown^T
    batch.add_op(HTP_OP_MUL_MAT, t_ffnout, {t_wdown, t_swiglu});

    // 14. Final residual: output = res1 + ffn_out  (written back to hidden slot)
    batch.add_op(HTP_OP_ADD, t_hidden, {t_res1, t_ffnout});

    // 15. Copy K and V to output slots (K already has RoPE applied)
    batch.add_op(HTP_OP_CPY, t_k_out, {t_k});
    batch.add_op(HTP_OP_CPY, t_v_out, {t_v});

    AEEResult err = flush_batch();
    if (err != AEE_SUCCESS) return false;

    hidden_tidx = t_hidden;
    return true;
}

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------
void FastRPCPrefill::Impl::teardown() {
    if (queue) {
        if (htp_handle) htp_iface_stop(htp_handle);
        dspqueue_close(queue);
        queue = nullptr;
    }
    if (htp_handle) {
        htp_iface_close(htp_handle);
        htp_handle = 0;
    }
    loaded = false;
}

// ---------------------------------------------------------------------------
// FastRPCPrefill public interface
// ---------------------------------------------------------------------------
FastRPCPrefill::FastRPCPrefill() : impl_(std::make_unique<Impl>()) {}
FastRPCPrefill::~FastRPCPrefill() { impl_->teardown(); }

bool FastRPCPrefill::is_available() const  { return impl_->loaded; }
int  FastRPCPrefill::get_chunk_size() const { return impl_->chunk_size; }
int  FastRPCPrefill::get_hidden_dim() const { return impl_->hidden_dim; }
int  FastRPCPrefill::get_num_layers() const { return impl_->num_layers; }
int  FastRPCPrefill::get_num_kv_heads() const { return impl_->num_kv_heads; }
int  FastRPCPrefill::get_head_dim() const   { return impl_->head_dim; }

bool FastRPCPrefill::load(const std::string& model_path) {
    auto& I = *impl_;
    I.teardown();

    // Initialize driver
    if (fastrpc_drv_init() != AEE_SUCCESS) {
        fprintf(stderr, "[FastRPC] driver init failed\n");
        return false;
    }

    // Query arch
    if (fastrpc_drv_get_arch(&I.arch) != 0) {
        fprintf(stderr, "[FastRPC] arch query failed, assuming v73\n");
        I.arch = 73;
    }
    fprintf(stderr, "[FastRPC] Hexagon arch: v%d\n", I.arch);

    // Parse model config
    if (!I.parse_config(model_path)) return false;

    // Allocate shared memory buffers
    if (!I.alloc_buffers()) return false;

    // Load layer weights into rpcmem
    I.layer_weights.resize(I.num_layers);
    for (int li = 0; li < I.num_layers; li++) {
        fprintf(stderr, "[FastRPC] loading layer %d / %d\n", li + 1, I.num_layers);
        if (!I.load_layer_weights(model_path, li)) return false;
    }

    // Open FastRPC session
    if (!I.open_session(I.arch)) return false;

    // Map all buffers into DSP address space
    if (!I.map_buffers()) return false;

    I.loaded = true;
    fprintf(stderr, "[FastRPC] model loaded: %s\n", model_path.c_str());
    return true;
}

NPUPrefillDirectResult FastRPCPrefill::prefill_chunk_direct(
        const std::vector<__fp16>& embeddings,
        int position_offset,
        const std::string& /*input_name*/) {

    NPUPrefillDirectResult result{};
    result.valid = false;

    auto& I = *impl_;
    if (!I.loaded) return result;

    int cs = I.chunk_size, em = I.hidden_dim;
    int nkv = I.num_kv_heads, hd = I.head_dim;

    // Copy embeddings to io_buf
    memcpy((uint8_t*)I.io_buf->base + I.io_off_emb,
           embeddings.data(),
           (size_t)cs * em * sizeof(uint16_t));

    // Build RoPE cos/sin tables
    I.build_rope_cos_sin(position_offset);

    // Run each layer
    uint16_t dummy = 0;
    for (int li = 0; li < I.num_layers; li++) {
        if (!I.run_layer(li, 0, 0, dummy)) return result;

        // After run_layer the updated hidden state is back in io_buf at io_off_emb
        // (the op graph writes the final residual there).
        // K/V for this layer are in io_buf at io_off_k[li] / io_off_v[li].
        size_t kv_elems = (size_t)cs * nkv * hd;
        memcpy(I.k_bufs[li].data(),
               (const uint8_t*)I.io_buf->base + I.io_off_k[li],
               kv_elems * 2);
        memcpy(I.v_bufs[li].data(),
               (const uint8_t*)I.io_buf->base + I.io_off_v[li],
               kv_elems * 2);
    }

    // Fill result
    result.hidden = { (const __fp16*)((const uint8_t*)I.io_buf->base + I.io_off_emb),
                      (size_t)cs * em };
    result.k_caches.resize(I.num_layers);
    result.v_caches.resize(I.num_layers);
    for (int li = 0; li < I.num_layers; li++) {
        result.k_caches[li] = { (const __fp16*)I.k_bufs[li].data(), I.k_bufs[li].size() };
        result.v_caches[li] = { (const __fp16*)I.v_bufs[li].data(), I.v_bufs[li].size() };
    }
    result.valid = true;
    return result;
}

} // namespace npu
} // namespace cactus
