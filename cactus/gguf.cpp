#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
struct type_to_lm_gguf_type;

template <>
struct type_to_lm_gguf_type<uint8_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_UINT8;
};

template <>
struct type_to_lm_gguf_type<int8_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_INT8;
};

template <>
struct type_to_lm_gguf_type<uint16_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_UINT16;
};

template <>
struct type_to_lm_gguf_type<int16_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_INT16;
};

template <>
struct type_to_lm_gguf_type<uint32_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_UINT32;
};

template <>
struct type_to_lm_gguf_type<int32_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_INT32;
};

template <>
struct type_to_lm_gguf_type<float> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_FLOAT32;
};

template <>
struct type_to_lm_gguf_type<bool> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_BOOL;
};

template <>
struct type_to_lm_gguf_type<std::string> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_STRING;
};

template <>
struct type_to_lm_gguf_type<uint64_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_UINT64;
};

template <>
struct type_to_lm_gguf_type<int64_t> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_INT64;
};

template <>
struct type_to_lm_gguf_type<double> {
    static constexpr enum lm_gguf_type value = LM_GGUF_TYPE_FLOAT64;
};

static const std::map<lm_gguf_type, size_t> LM_GGUF_TYPE_SIZE = {
    {LM_GGUF_TYPE_UINT8,   sizeof(uint8_t)},
    {LM_GGUF_TYPE_INT8,    sizeof(int8_t)},
    {LM_GGUF_TYPE_UINT16,  sizeof(uint16_t)},
    {LM_GGUF_TYPE_INT16,   sizeof(int16_t)},
    {LM_GGUF_TYPE_UINT32,  sizeof(uint32_t)},
    {LM_GGUF_TYPE_INT32,   sizeof(int32_t)},
    {LM_GGUF_TYPE_FLOAT32, sizeof(float)},
    {LM_GGUF_TYPE_BOOL,    sizeof(int8_t)},
    {LM_GGUF_TYPE_STRING,  0}, // undefined
    {LM_GGUF_TYPE_ARRAY,   0}, // undefined
    {LM_GGUF_TYPE_UINT64,  sizeof(uint64_t)},
    {LM_GGUF_TYPE_INT64,   sizeof(int64_t)},
    {LM_GGUF_TYPE_FLOAT64, sizeof(double)},
};
static_assert(LM_GGUF_TYPE_COUNT == 13, "LM_GGUF_TYPE_COUNT != 13");

static const std::map<lm_gguf_type, const char *> LM_GGUF_TYPE_NAME = {
    {LM_GGUF_TYPE_UINT8,   "u8"},
    {LM_GGUF_TYPE_INT8,    "i8"},
    {LM_GGUF_TYPE_UINT16,  "u16"},
    {LM_GGUF_TYPE_INT16,   "i16"},
    {LM_GGUF_TYPE_UINT32,  "u32"},
    {LM_GGUF_TYPE_INT32,   "i32"},
    {LM_GGUF_TYPE_FLOAT32, "f32"},
    {LM_GGUF_TYPE_BOOL,    "bool"},
    {LM_GGUF_TYPE_STRING,  "str"},
    {LM_GGUF_TYPE_ARRAY,   "arr"},
    {LM_GGUF_TYPE_UINT64,  "u64"},
    {LM_GGUF_TYPE_INT64,   "i64"},
    {LM_GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(LM_GGUF_TYPE_COUNT == 13, "LM_GGUF_TYPE_COUNT != 13");

size_t lm_gguf_type_size(enum lm_gguf_type type) {
    auto it = LM_GGUF_TYPE_SIZE.find(type);
    return it == LM_GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

struct lm_gguf_kv {
    std::string key;

    bool is_array;
    enum lm_gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    lm_gguf_kv(const std::string & key, const T value)
            : key(key), is_array(false), type(type_to_lm_gguf_type<T>::value) {
        LM_GGML_ASSERT(!key.empty());
        data.resize(sizeof(T));
        memcpy(data.data(), &value, sizeof(T));
    }

    template <typename T>
    lm_gguf_kv(const std::string & key, const std::vector<T> & value)
            : key(key), is_array(true), type(type_to_lm_gguf_type<T>::value) {
        LM_GGML_ASSERT(!key.empty());
        data.resize(value.size()*sizeof(T));
        for (size_t i = 0; i < value.size(); ++i) {
            const T tmp = value[i];
            memcpy(data.data() + i*sizeof(T), &tmp, sizeof(T));
        }
    }

    lm_gguf_kv(const std::string & key, const std::string & value)
            : key(key), is_array(false), type(LM_GGUF_TYPE_STRING) {
        LM_GGML_ASSERT(!key.empty());
        data_string.push_back(value);
    }

    lm_gguf_kv(const std::string & key, const std::vector<std::string> & value)
            : key(key), is_array(true), type(LM_GGUF_TYPE_STRING) {
        LM_GGML_ASSERT(!key.empty());
        data_string = value;
    }

    const std::string & get_key() const {
        return key;
    }

    const enum lm_gguf_type & get_type() const {
        return type;
    }

    size_t get_ne() const {
        if (type == LM_GGUF_TYPE_STRING) {
            const size_t ne = data_string.size();
            LM_GGML_ASSERT(is_array || ne == 1);
            return ne;
        }
        const size_t type_size = lm_gguf_type_size(type);
        LM_GGML_ASSERT(data.size() % type_size == 0);
        const size_t ne = data.size() / type_size;
        LM_GGML_ASSERT(is_array || ne == 1);
        return ne;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        LM_GGML_ASSERT(type_to_lm_gguf_type<T>::value == type);
        if constexpr (std::is_same<T, std::string>::value) {
            LM_GGML_ASSERT(data_string.size() >= i+1);
            return data_string[i];
        }
        const size_t type_size = lm_gguf_type_size(type);
        LM_GGML_ASSERT(data.size() % type_size == 0);
        LM_GGML_ASSERT(data.size() >= (i+1)*type_size);
        return reinterpret_cast<const T *>(data.data())[i];
    }

    void cast(const enum lm_gguf_type new_type) {
        const size_t new_type_size = lm_gguf_type_size(new_type);
        LM_GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

struct lm_gguf_tensor_info {
    struct lm_ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct lm_gguf_context {
    uint32_t version = LM_GGUF_VERSION;

    std::vector<struct lm_gguf_kv> kv;
    std::vector<struct lm_gguf_tensor_info> info;

    size_t alignment = LM_GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    void * data = nullptr;
};

struct lm_gguf_reader {
    FILE * file;

    lm_gguf_reader(FILE * file) : file(file) {}

    template <typename T>
    bool read(T & dst) const {
        return fread(&dst, 1, sizeof(dst), file) == sizeof(dst);
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            } else {
                if (!read(dst[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool & dst) const {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum lm_ggml_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = lm_ggml_type(tmp);
        return true;
    }

    bool read(enum lm_gguf_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = lm_gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) const {
        uint64_t size = -1;
        if (!read(size)) {
            return false;
        }
        dst.resize(size);
        return fread(dst.data(), 1, dst.length(), file) == dst.length();
    }

    bool read(void * dst, const size_t size) const {
        return fread(dst, 1, size, file) == size;
    }
};

struct lm_gguf_context * lm_gguf_init_empty(void) {
    return new lm_gguf_context;
}

template<typename T>
bool lm_gguf_read_emplace_helper(const struct lm_gguf_reader & gr, std::vector<struct lm_gguf_kv> & kv, const std::string & key, const bool is_array, const size_t n) {
    if (is_array) {
        std::vector<T> value;
        try {
            if (!gr.read(value, n)) {
                return false;
            }
        } catch (std::length_error &) {
            LM_GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        } catch (std::bad_alloc &) {
            LM_GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        }
        kv.emplace_back(key, value);
    } else {
        T value;
        if (!gr.read(value)) {
            return false;
        }
        kv.emplace_back(key, value);
    }
    return true;
}

struct lm_gguf_context * lm_gguf_init_from_file_impl(FILE * file, struct lm_gguf_init_params params) {
    const struct lm_gguf_reader gr(file);
    struct lm_gguf_context * ctx = new lm_gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            LM_GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            lm_gguf_free(ctx);
            return nullptr;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != LM_GGUF_MAGIC[i]) {
                LM_GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                lm_gguf_free(ctx);
                return nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    if (ok && gr.read(ctx->version)) {
        if (ctx->version == 1) {
            LM_GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        if (ctx->version > LM_GGUF_VERSION) {
            LM_GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, LM_GGUF_VERSION);
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_tensors)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(lm_gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(lm_gguf_tensor_info))) {
            LM_GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/sizeof(lm_gguf_tensor_info));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(lm_gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(lm_gguf_kv))) {
            LM_GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/sizeof(lm_gguf_kv));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (!ok) {
        LM_GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        lm_gguf_free(ctx);
        return nullptr;
    }

    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            lm_gguf_type   type     = lm_gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } catch (std::length_error &) {
                LM_GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                LM_GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    LM_GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == LM_GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            if (!ok) {
                break;
            }

            switch (type) {
                case LM_GGUF_TYPE_UINT8:   ok = ok && lm_gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_INT8:    ok = ok && lm_gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_UINT16:  ok = ok && lm_gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_INT16:   ok = ok && lm_gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_UINT32:  ok = ok && lm_gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_INT32:   ok = ok && lm_gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_FLOAT32: ok = ok && lm_gguf_read_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_BOOL:    ok = ok && lm_gguf_read_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_STRING:  ok = ok && lm_gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_UINT64:  ok = ok && lm_gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_INT64:   ok = ok && lm_gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_FLOAT64: ok = ok && lm_gguf_read_emplace_helper<double>     (gr, ctx->kv, key, is_array, n); break;
                case LM_GGUF_TYPE_ARRAY:
                default:
                    {
                        LM_GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            LM_GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            lm_gguf_free(ctx);
            return nullptr;
        }
        LM_GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        const int alignment_idx = lm_gguf_find_key(ctx, LM_GGUF_KEY_GENERAL_ALIGNMENT);
        ctx->alignment = alignment_idx == -1 ? LM_GGUF_DEFAULT_ALIGNMENT : lm_gguf_get_val_u32(ctx, alignment_idx);

        if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            LM_GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            lm_gguf_free(ctx);
            return nullptr;
        }
    }

    // read the tensor info
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        struct lm_gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            } catch (std::length_error &) {
                LM_GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                LM_GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            if (name.length() >= LM_GGML_MAX_NAME) {
                LM_GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), LM_GGML_MAX_NAME);
                ok = false;
                break;
            }
            lm_ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    LM_GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = -1;
            ok = ok && gr.read(n_dims);
            if (n_dims > LM_GGML_MAX_DIMS) {
                LM_GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, LM_GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < LM_GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                if (info.t.ne[j] < 0) {
                    LM_GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            // check that the total number of elements is representable
            if (ok && ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                       (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                       (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {

                LM_GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                    "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= LM_GGML_TYPE_COUNT) {
                LM_GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d (%s)\n",
                    __func__, info.t.name, info.t.type, lm_ggml_type_name(info.t.type));
                ok = false;
                break;
            }
            const size_t  type_size = lm_ggml_type_size(info.t.type);
            const int64_t blck_size = lm_ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                LM_GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, lm_ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < LM_GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    if (!ok) {
        LM_GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        lm_gguf_free(ctx);
        return nullptr;
    }
    LM_GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if (fseek(file, LM_GGML_PAD(ftell(file), ctx->alignment), SEEK_SET) != 0) {
        LM_GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        lm_gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = ftell(file);

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const lm_gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                LM_GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                LM_GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                lm_gguf_free(ctx);
                return nullptr;
            }
            ctx->size += LM_GGML_PAD(lm_ggml_nbytes(&ti.t), ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != nullptr) {
        // if the provided lm_gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created lm_ggml_context as well, and point the "data" members of
        //   the lm_ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new lm_ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (n_tensors    )*lm_ggml_tensor_overhead() :
            (n_tensors + 1)*lm_ggml_tensor_overhead() + ctx->size;

        struct lm_ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = lm_ggml_init(pdata);
        if (*params.ctx == nullptr) {
            LM_GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            lm_gguf_free(ctx);
            return nullptr;
        }

        struct lm_ggml_context * ctx_data = *params.ctx;

        struct lm_ggml_tensor * data = nullptr;

        if (!params.no_alloc) {
            data = lm_ggml_new_tensor_1d(ctx_data, LM_GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            if (ok) {
                lm_ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.read(data->data, ctx->size);

            if (!ok) {
                LM_GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                lm_ggml_free(ctx_data);
                *params.ctx = nullptr;
                lm_gguf_free(ctx);
                return nullptr;
            }

            ctx->data = data->data;
        }

        lm_ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const struct lm_gguf_tensor_info & info = ctx->info[i];

            struct lm_ggml_tensor * cur = lm_ggml_new_tensor(ctx_data, info.t.type, LM_GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            if (!ok) {
                break;
            }

            lm_ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            if (!params.no_alloc) {
                cur->data = (char *) data->data + info.offset;
            }
        }

        if (!ok) {
            LM_GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            lm_ggml_free(ctx_data);
            *params.ctx = nullptr;
            lm_gguf_free(ctx);
            return nullptr;
        }

        lm_ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}

struct lm_gguf_context * lm_gguf_init_from_file(const char * fname, struct lm_gguf_init_params params) {
    FILE * file = lm_ggml_fopen(fname, "rb");

    if (!file) {
        LM_GGML_LOG_ERROR("%s: failed to open GGUF file '%s'\n", __func__, fname);
        return nullptr;
    }

    struct lm_gguf_context * result = lm_gguf_init_from_file_impl(file, params);
    fclose(file);
    return result;
}

void lm_gguf_free(struct lm_gguf_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

const char * lm_gguf_type_name(enum lm_gguf_type type) {
    auto it = LM_GGUF_TYPE_NAME.find(type);
    return it == LM_GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t lm_gguf_get_version(const struct lm_gguf_context * ctx) {
    return ctx->version;
}

size_t lm_gguf_get_alignment(const struct lm_gguf_context * ctx) {
    return ctx->alignment;
}

size_t lm_gguf_get_data_offset(const struct lm_gguf_context * ctx) {
    return ctx->offset;
}

int64_t lm_gguf_get_n_kv(const struct lm_gguf_context * ctx) {
    return ctx->kv.size();
}

int64_t lm_gguf_find_key(const struct lm_gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int64_t keyfound = -1;

    const int64_t n_kv = lm_gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        if (strcmp(key, lm_gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * lm_gguf_get_key(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].get_key().c_str();
}

enum lm_gguf_type lm_gguf_get_kv_type(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].is_array ? LM_GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
}

enum lm_gguf_type lm_gguf_get_arr_type(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].is_array);
    return ctx->kv[key_id].get_type();
}

const void * lm_gguf_get_arr_data(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_type() != LM_GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

const char * lm_gguf_get_arr_str(const struct lm_gguf_context * ctx, int64_t key_id, size_t i) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_type() == LM_GGUF_TYPE_STRING);
    return ctx->kv[key_id].data_string[i].c_str();
}

size_t lm_gguf_get_arr_n(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));

    if (ctx->kv[key_id].type == LM_GGUF_TYPE_STRING) {
        return ctx->kv[key_id].data_string.size();
    }

    const size_t type_size = lm_gguf_type_size(ctx->kv[key_id].type);
    LM_GGML_ASSERT(ctx->kv[key_id].data.size() % type_size == 0);
    return ctx->kv[key_id].data.size() / type_size;
}

uint8_t lm_gguf_get_val_u8(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint8_t>();
}

int8_t lm_gguf_get_val_i8(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int8_t>();
}

uint16_t lm_gguf_get_val_u16(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint16_t>();
}

int16_t lm_gguf_get_val_i16(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int16_t>();
}

uint32_t lm_gguf_get_val_u32(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint32_t>();
}

int32_t lm_gguf_get_val_i32(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int32_t>();
}

float lm_gguf_get_val_f32(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<float>();
}

uint64_t lm_gguf_get_val_u64(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint64_t>();
}

int64_t lm_gguf_get_val_i64(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int64_t>();
}

double lm_gguf_get_val_f64(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<double>();
}

bool lm_gguf_get_val_bool(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<bool>();
}

const char * lm_gguf_get_val_str(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<std::string>().c_str();
}

const void * lm_gguf_get_val_data(const struct lm_gguf_context * ctx, int64_t key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    LM_GGML_ASSERT(ctx->kv[key_id].get_type() != LM_GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

int64_t lm_gguf_get_n_tensors(const struct lm_gguf_context * ctx) {
    return ctx->info.size();
}

int64_t lm_gguf_find_tensor(const struct lm_gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    const int64_t n_tensors = lm_gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; ++i) {
        if (strcmp(name, lm_gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            break;
        }
    }

    return tensor_id;
}

size_t lm_gguf_get_tensor_offset(const struct lm_gguf_context * ctx, int64_t tensor_id) {
    LM_GGML_ASSERT(tensor_id >= 0 && tensor_id < lm_gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].offset;
}

const char * lm_gguf_get_tensor_name(const struct lm_gguf_context * ctx, int64_t tensor_id) {
    LM_GGML_ASSERT(tensor_id >= 0 && tensor_id < lm_gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.name;
}

enum lm_ggml_type lm_gguf_get_tensor_type(const struct lm_gguf_context * ctx, int64_t tensor_id) {
    LM_GGML_ASSERT(tensor_id >= 0 && tensor_id < lm_gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.type;
}

size_t lm_gguf_get_tensor_size(const struct lm_gguf_context * ctx, int64_t tensor_id) {
    LM_GGML_ASSERT(tensor_id >= 0 && tensor_id < lm_gguf_get_n_tensors(ctx));
    return lm_ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t lm_gguf_remove_key(struct lm_gguf_context * ctx, const char * key) {
    const int64_t key_id = lm_gguf_find_key(ctx, key);
    if (key_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + key_id);
    }
    return key_id;
}

template<typename T>
static void lm_gguf_check_reserved_keys(const std::string & key, const T val) {
    if (key == LM_GGUF_KEY_GENERAL_ALIGNMENT) {
        if constexpr (std::is_same<T, uint32_t>::value) {
            LM_GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && LM_GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
        } else {
            LM_GGML_UNUSED(val);
            LM_GGML_ABORT(LM_GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

void lm_gguf_set_val_u8(struct lm_gguf_context * ctx, const char * key, uint8_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_i8(struct lm_gguf_context * ctx, const char * key, int8_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_u16(struct lm_gguf_context * ctx, const char * key, uint16_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_i16(struct lm_gguf_context * ctx, const char * key, int16_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_u32(struct lm_gguf_context * ctx, const char * key, uint32_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_i32(struct lm_gguf_context * ctx, const char * key, int32_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_f32(struct lm_gguf_context * ctx, const char * key, float val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_u64(struct lm_gguf_context * ctx, const char * key, uint64_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_i64(struct lm_gguf_context * ctx, const char * key, int64_t val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_f64(struct lm_gguf_context * ctx, const char * key, double val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_bool(struct lm_gguf_context * ctx, const char * key, bool val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void lm_gguf_set_val_str(struct lm_gguf_context * ctx, const char * key, const char * val) {
    lm_gguf_check_reserved_keys(key, val);
    lm_gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

void lm_gguf_set_arr_data(struct lm_gguf_context * ctx, const char * key, enum lm_gguf_type type, const void * data, size_t n) {
    lm_gguf_check_reserved_keys(key, data);
    lm_gguf_remove_key(ctx, key);

    const size_t nbytes = n*lm_gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    if (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().cast(type);
}

void lm_gguf_set_arr_str(struct lm_gguf_context * ctx, const char * key, const char ** data, size_t n) {
    lm_gguf_check_reserved_keys(key, data);
    lm_gguf_remove_key(ctx, key);

    std::vector<std::string> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
void lm_gguf_set_kv(struct lm_gguf_context * ctx, const struct lm_gguf_context * src) {
    const int64_t n_kv = lm_gguf_get_n_kv(src);
    for (int64_t i = 0; i < n_kv; ++i) {
        const struct lm_gguf_kv & kv = src->kv[i];

        if (!kv.is_array) {
            switch (kv.get_type()) {
                case LM_GGUF_TYPE_UINT8:   lm_gguf_set_val_u8  (ctx, kv.get_key().c_str(), kv.get_val<uint8_t>());             break;
                case LM_GGUF_TYPE_INT8:    lm_gguf_set_val_i8  (ctx, kv.get_key().c_str(), kv.get_val<int8_t>());              break;
                case LM_GGUF_TYPE_UINT16:  lm_gguf_set_val_u16 (ctx, kv.get_key().c_str(), kv.get_val<uint16_t>());            break;
                case LM_GGUF_TYPE_INT16:   lm_gguf_set_val_i16 (ctx, kv.get_key().c_str(), kv.get_val<int16_t>());             break;
                case LM_GGUF_TYPE_UINT32:  lm_gguf_set_val_u32 (ctx, kv.get_key().c_str(), kv.get_val<uint32_t>());            break;
                case LM_GGUF_TYPE_INT32:   lm_gguf_set_val_i32 (ctx, kv.get_key().c_str(), kv.get_val<int32_t>());             break;
                case LM_GGUF_TYPE_FLOAT32: lm_gguf_set_val_f32 (ctx, kv.get_key().c_str(), kv.get_val<float>());               break;
                case LM_GGUF_TYPE_UINT64:  lm_gguf_set_val_u64 (ctx, kv.get_key().c_str(), kv.get_val<uint64_t>());            break;
                case LM_GGUF_TYPE_INT64:   lm_gguf_set_val_i64 (ctx, kv.get_key().c_str(), kv.get_val<int64_t>());             break;
                case LM_GGUF_TYPE_FLOAT64: lm_gguf_set_val_f64 (ctx, kv.get_key().c_str(), kv.get_val<double>());              break;
                case LM_GGUF_TYPE_BOOL:    lm_gguf_set_val_bool(ctx, kv.get_key().c_str(), kv.get_val<bool>());                break;
                case LM_GGUF_TYPE_STRING:  lm_gguf_set_val_str (ctx, kv.get_key().c_str(), kv.get_val<std::string>().c_str()); break;
                case LM_GGUF_TYPE_ARRAY:
                default: LM_GGML_ABORT("invalid type");
            }
            continue;
        }

        const size_t ne = kv.get_ne();

        switch (kv.get_type()) {
            case LM_GGUF_TYPE_UINT8:
            case LM_GGUF_TYPE_INT8:
            case LM_GGUF_TYPE_UINT16:
            case LM_GGUF_TYPE_INT16:
            case LM_GGUF_TYPE_UINT32:
            case LM_GGUF_TYPE_INT32:
            case LM_GGUF_TYPE_FLOAT32:
            case LM_GGUF_TYPE_UINT64:
            case LM_GGUF_TYPE_INT64:
            case LM_GGUF_TYPE_FLOAT64:
            case LM_GGUF_TYPE_BOOL: {
                lm_gguf_set_arr_data(ctx, kv.get_key().c_str(), kv.get_type(), kv.data.data(), ne);
            } break;
            case LM_GGUF_TYPE_STRING: {
                std::vector<const char *> tmp(ne);
                for (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                lm_gguf_set_arr_str(ctx, kv.get_key().c_str(), tmp.data(), ne);
            } break;
            case LM_GGUF_TYPE_ARRAY:
            default: LM_GGML_ABORT("invalid type");
        }
    }
}

void lm_gguf_add_tensor(
             struct lm_gguf_context * ctx,
        const struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);
    if (lm_gguf_find_tensor(ctx, tensor->name) != -1) {
        LM_GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    struct lm_gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + LM_GGML_PAD(lm_ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

void lm_gguf_set_tensor_type(struct lm_gguf_context * ctx, const char * name, enum lm_ggml_type type) {
    const int64_t tensor_id = lm_gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        LM_GGML_ABORT("tensor not found: %s", name);
    }
    struct lm_ggml_tensor * tensor = &ctx->info[tensor_id].t;
    const size_t  type_size = lm_ggml_type_size(type);
    const int64_t blck_size = lm_ggml_blck_size(type);

    tensor->type = type;
    LM_GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    for (int i = 2; i < LM_GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    const int64_t n_tensors = lm_gguf_get_n_tensors(ctx);
    for (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + LM_GGML_PAD(lm_ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

void lm_gguf_set_tensor_data(struct lm_gguf_context * ctx, const char * name, const void * data) {
    const int64_t tensor_id = lm_gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        LM_GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

struct lm_gguf_writer {
    std::vector<int8_t> & buf;

    lm_gguf_writer(std::vector<int8_t> & buf) : buf(buf) {}

    template <typename T>
    void write(const T & val) const {
        for (size_t i = 0; i < sizeof(val); ++i) {
            buf.push_back(reinterpret_cast<const int8_t *>(&val)[i]);
        }
    }

    void write(const std::vector<int8_t> & val) const {
        buf.insert(buf.end(), val.begin(), val.end());
    }

    void write(const bool & val) const {
        const int8_t val8 = val ? 1 : 0;
        write(val8);
    }

    void write(const std::string & val) const {
        {
            const uint64_t n = val.length();
            write(n);
        }
        for (size_t i = 0; i < val.length(); ++i) {
            buf.push_back(reinterpret_cast<const int8_t *>(val.data())[i]);
        }
    }

    void write(const char * val) const {
        write(std::string(val));
    }

    void write(const enum lm_ggml_type & val) const {
        write(int32_t(val));
    }

    void write(const enum lm_gguf_type & val) const {
        write(int32_t(val));
    }

    void write(const struct lm_gguf_kv & kv) const {
        const uint64_t ne = kv.get_ne();

        write(kv.get_key());

        if (kv.is_array) {
            write(LM_GGUF_TYPE_ARRAY);
            write(kv.get_type());
            write(ne);
        } else {
            write(kv.get_type());
        }

        switch (kv.get_type()) {
            case LM_GGUF_TYPE_UINT8:
            case LM_GGUF_TYPE_INT8:
            case LM_GGUF_TYPE_UINT16:
            case LM_GGUF_TYPE_INT16:
            case LM_GGUF_TYPE_UINT32:
            case LM_GGUF_TYPE_INT32:
            case LM_GGUF_TYPE_FLOAT32:
            case LM_GGUF_TYPE_UINT64:
            case LM_GGUF_TYPE_INT64:
            case LM_GGUF_TYPE_FLOAT64: {
                write(kv.data);
            } break;
            case LM_GGUF_TYPE_BOOL: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<bool>(i));
                }
            } break;
            case LM_GGUF_TYPE_STRING: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<std::string>(i));
                }
            } break;
            case LM_GGUF_TYPE_ARRAY:
            default: LM_GGML_ABORT("invalid type");
        }
    }

    void write_tensor_meta(const struct lm_gguf_tensor_info & info) const {
        write(info.t.name);

        const uint32_t n_dims = lm_ggml_n_dims(&info.t);
        write(n_dims);

        for (uint32_t j = 0; j < n_dims; ++j) {
            write(info.t.ne[j]);
        }
        write(info.t.type);
        write(info.offset);
    }

    void pad(const size_t alignment) const {
        while (buf.size() % alignment != 0) {
            const int8_t zero = 0;
            write(zero);
        }
    }

    void write_tensor_data(const struct lm_gguf_tensor_info & info, const size_t offset_data, const size_t alignment) const {
        LM_GGML_ASSERT(buf.size() - offset_data == info.offset);

        LM_GGML_ASSERT(lm_ggml_is_contiguous(&info.t));
        const size_t offset = buf.size();
        const size_t nbytes = lm_ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        if (info.t.buffer) {
            lm_ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } else {
            LM_GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }

        pad(alignment);
    }
};

void lm_gguf_write_to_buf(const struct lm_gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta) {
    const struct lm_gguf_writer gw(buf);

    const int64_t n_kv      = lm_gguf_get_n_kv(ctx);
    const int64_t n_tensors = lm_gguf_get_n_tensors(ctx);

    // write header
    gw.write(LM_GGUF_MAGIC[0]);
    gw.write(LM_GGUF_MAGIC[1]);
    gw.write(LM_GGUF_MAGIC[2]);
    gw.write(LM_GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }

    // write tensor info
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    if (only_meta) {
        return;
    }

    const size_t offset_data = gw.buf.size();

    // write tensor data
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

bool lm_gguf_write_to_file(const struct lm_gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = lm_ggml_fopen(fname, "wb");

    if (!file) {
        LM_GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        return false;
    }

    std::vector<int8_t> buf;
    lm_gguf_write_to_buf(ctx, buf, only_meta);
    const bool ok = fwrite(buf.data(), 1, buf.size(), file) == buf.size();
    fclose(file);
    return ok;
}

size_t lm_gguf_get_meta_size(const struct lm_gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    lm_gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    return buf.size();
}

void lm_gguf_get_meta_data(const struct lm_gguf_context * ctx, void * data) {
    std::vector<int8_t> buf;
    lm_gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    memcpy(data, buf.data(), buf.size());
}
