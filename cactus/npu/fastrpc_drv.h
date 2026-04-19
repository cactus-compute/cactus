#pragma once

// FastRPC driver interface for Snapdragon Windows.
// Loads libcdsprpc.dll from the Qualcomm driver store and exposes the
// rpcmem / dspqueue / remote_handle64 APIs needed by the HTP backend.
//
// All SDK types that would normally come from Hexagon SDK headers are
// redefined here so the host build requires no SDK installation.

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

// ---------------------------------------------------------------------------
// AEEResult / error codes (from AEEStdErr.h)
// ---------------------------------------------------------------------------
typedef int AEEResult;
#define AEE_SUCCESS           0
#define AEE_EOFFSET           0x80000400u
#define AEE_EUNABLETOLOAD     (AEE_EOFFSET + 0x006)
#define AEE_EUNSUPPORTEDAPI   (AEE_EOFFSET + 0x0E7)
#define AEE_ECONNRESET        104

// ---------------------------------------------------------------------------
// remote_handle64 / remote_arg (from remote.h)
// ---------------------------------------------------------------------------
typedef uint64_t remote_handle64;
typedef uint32_t remote_handle;

struct remote_buf  { void* pv; size_t nLen; };
struct remote_dma_handle { int32_t fd; uint32_t len; uint32_t offset; };
struct remote_arg {
    union {
        struct remote_buf       buf;
        remote_handle           h;
        struct remote_dma_handle dma;
    };
};

// Scalar encoding helpers (matches Hexagon IDL calling convention)
#define REMOTE_SCALARS_MAKEX(ctx,mid,nIn,nROut,noIn,noROut) \
    (uint32_t)((((ctx)&0xff)<<24)|(((mid)&0xff)<<16)|(((nIn)&0xff)<<12)| \
               (((nROut)&0xff)<<8)|(((noIn)&0xf)<<4)|((noROut)&0xf))

// Control request codes (from remote.h, verified against Hexagon SDK 6.4.0.2)
// handle_control_req_id enum:
#define DSPRPC_GET_DSP_INFO              2   // remote_handle_control()
#define DSPRPC_CONTROL_LATENCY           3   // remote_handle64_control() — QoS
// session_control_req_id enum:
#define DSPRPC_CONTROL_UNSIGNED_MODULE   2   // remote_session_control()
#define FASTRPC_RESERVE_NEW_SESSION     13   // remote_session_control()
#define FASTRPC_GET_URI                 15   // remote_session_control()

struct remote_rpc_control_latency {
    uint32_t enable;
};

// DSP capability attribute IDs (dsp_capability_attribute enum in remote.h)
#define ARCH_VER 6  // Hexagon processor architecture version (6th entry in remote_dsp_attributes)

struct remote_dsp_capability {
    uint32_t domain;
    uint32_t attribute_ID;
    uint32_t capability;
};

struct remote_rpc_control_unsigned_module {
    int domain;
    int enable;
};

struct remote_rpc_reserve_new_session {
    char*    domain_name;
    uint32_t domain_name_len;
    char*    session_name;
    uint32_t session_name_len;
    // Filled in by kernel (effective_domain_id comes before session_id per SDK):
    uint32_t effective_domain_id;
    uint32_t session_id;
};

struct remote_rpc_get_uri {
    // Field order matches SDK remote_rpc_get_uri_t exactly:
    char*    domain_name;
    uint32_t domain_name_len;
    uint32_t session_id;
    char*    module_uri;
    uint32_t module_uri_len;
    char*    uri;
    uint32_t uri_len;
};

#define CDSP_DOMAIN_NAME "&_dom=cdsp"
#define MAX_DOMAIN_NAMELEN 64

// ---------------------------------------------------------------------------
// rpcmem heap / flags (from rpcmem.h)
// ---------------------------------------------------------------------------
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS  1

// ---------------------------------------------------------------------------
// fastrpc_map_flags (from fastrpc.h)
// ---------------------------------------------------------------------------
enum fastrpc_map_flags {
    FASTRPC_MAP_STATIC      = 0,
    FASTRPC_MAP_MAX         = 0,
};

// ---------------------------------------------------------------------------
// dspqueue types (from dspqueue.h)
// ---------------------------------------------------------------------------
typedef void* dspqueue_t;
typedef void (*dspqueue_callback_t)(dspqueue_t queue, int error, void* context);

#define DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER       (1u << 0)
#define DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT (1u << 1)
#define DSPQUEUE_TIMEOUT_NONE ((uint32_t)-1)
#define DSPQUEUE_TIMEOUT      DSPQUEUE_TIMEOUT_NONE

struct dspqueue_buffer {
    uint32_t flags;
    int32_t  fd;
    uint32_t offset;
    uint32_t size;
    void*    ptr;
};

// ---------------------------------------------------------------------------
// Driver init / query
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

// Initialize the FastRPC driver (load libcdsprpc.dll from driver store).
// Returns AEE_SUCCESS on success.
AEEResult fastrpc_drv_init(void);

// Query the Hexagon arch version (e.g. 73 for v73).
int fastrpc_drv_get_arch(int* arch_out);

// ---------------------------------------------------------------------------
// rpcmem API (forwarded from libcdsprpc.dll)
// ---------------------------------------------------------------------------
void*  rpcmem_alloc(int heapid, uint32_t flags, int size);
void   rpcmem_free(void* p);
int    rpcmem_to_fd(void* p);

// ---------------------------------------------------------------------------
// fastrpc mmap/munmap
// ---------------------------------------------------------------------------
int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length,
                 enum fastrpc_map_flags flags);
int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

// ---------------------------------------------------------------------------
// dspqueue API
// ---------------------------------------------------------------------------
AEEResult dspqueue_create(int domain, uint32_t flags,
                          uint32_t req_queue_size, uint32_t resp_queue_size,
                          dspqueue_callback_t packet_cb, dspqueue_callback_t error_cb,
                          void* cb_context, dspqueue_t* queue_out);
AEEResult dspqueue_close(dspqueue_t queue);
AEEResult dspqueue_export(dspqueue_t queue, uint64_t* queue_id_out);
AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags,
                         uint32_t n_bufs, struct dspqueue_buffer* bufs,
                         uint32_t msg_len, const uint8_t* msg,
                         uint32_t timeout_us);
AEEResult dspqueue_read(dspqueue_t queue, uint32_t* flags_out,
                        uint32_t max_bufs, uint32_t* n_bufs_out,
                        struct dspqueue_buffer* bufs,
                        uint32_t max_msg_len, uint32_t* msg_len_out,
                        uint8_t* msg, uint32_t timeout_us);

// ---------------------------------------------------------------------------
// remote_handle64 API
// ---------------------------------------------------------------------------
int remote_handle64_open(const char* uri, remote_handle64* h_out);
int remote_handle64_close(remote_handle64 h);
int remote_handle64_invoke(remote_handle64 h, uint32_t scalars, struct remote_arg* pra);
int remote_handle_control(uint32_t req, void* data, uint32_t datalen);
int remote_handle64_control(remote_handle64 h, uint32_t req, void* data, uint32_t datalen);
int remote_session_control(uint32_t req, void* data, uint32_t datalen);

// ---------------------------------------------------------------------------
// htp_iface IDL stub (inlined — no separate IDL build needed)
// Wraps remote_handle64_invoke with the correct scalars / buffer layout.
// Method IDs match those in htp_iface.idl compiled by the Hexagon IDL compiler.
// ---------------------------------------------------------------------------
static inline AEEResult htp_iface_open(const char* uri, remote_handle64* h) {
    return remote_handle64_open(uri, h);
}
static inline AEEResult htp_iface_close(remote_handle64 h) {
    return remote_handle64_close(h);
}
// start(sess_id: uint32, dsp_queue_id: uint64, n_hvx: uint32, use_hmx: uint32)
static inline AEEResult htp_iface_start(remote_handle64 h,
                                        uint32_t sess_id, uint64_t dsp_queue_id,
                                        uint32_t n_hvx, uint32_t use_hmx) {
    struct remote_arg pra[1] = {};
    uint64_t primIn[3] = {};
    memcpy(&primIn[0], &sess_id,      4);
    memcpy(&primIn[1], &dsp_queue_id, 8);
    // n_hvx at byte 16, use_hmx at byte 20
    uint32_t rest[2] = { n_hvx, use_hmx };
    memcpy((char*)primIn + 16, rest, 8);
    pra[0].buf.pv   = primIn;
    pra[0].buf.nLen = sizeof(primIn);
    // method index 2, 1 input buffer
    return remote_handle64_invoke(h, REMOTE_SCALARS_MAKEX(0, 2, 1, 0, 0, 0), pra);
}
// stop()
static inline AEEResult htp_iface_stop(remote_handle64 h) {
    // method index 3, no buffers — pass empty pra array (DLL crashes on null pra)
    struct remote_arg pra[1] = {};
    return remote_handle64_invoke(h, REMOTE_SCALARS_MAKEX(0, 3, 0, 0, 0, 0), pra);
}
// mmap(fd: uint32, size: uint32, pinned: uint32)
static inline AEEResult htp_iface_mmap(remote_handle64 h,
                                        uint32_t fd, uint32_t size, uint32_t pinned) {
    struct remote_arg pra[1] = {};
    uint32_t primIn[3] = { fd, size, pinned };
    pra[0].buf.pv   = primIn;
    pra[0].buf.nLen = sizeof(primIn);
    return remote_handle64_invoke(h, REMOTE_SCALARS_MAKEX(0, 4, 1, 0, 0, 0), pra);
}
// munmap(fd: uint32)
static inline AEEResult htp_iface_munmap(remote_handle64 h, uint32_t fd) {
    struct remote_arg pra[1] = {};
    uint32_t primIn[1] = { fd };
    pra[0].buf.pv   = primIn;
    pra[0].buf.nLen = sizeof(primIn);
    return remote_handle64_invoke(h, REMOTE_SCALARS_MAKEX(0, 5, 1, 0, 0, 0), pra);
}

#ifdef __cplusplus
}
#endif
