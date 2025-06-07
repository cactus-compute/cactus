#include "llama-model-loader.h"

#include "ggml.h"

#include <array>
#include <cinttypes>
#include <cstring>
#include <future>

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case LM_GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case LM_GGUF_FILE_VERSION_V2: return "GGUF V2";
        case LM_GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_TQ1_0:    return "TQ1_0 - 1.69 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_TQ2_0:    return "TQ2_0 - 2.06 bpw ternary";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";

        default: return "unknown, may not work";
    }
}

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

namespace GGUFMeta {
    template <typename T, lm_gguf_type gt_, T (*gfun)(const lm_gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr lm_gguf_type gt = gt_;

        static T getter(const lm_gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         LM_GGUF_TYPE_BOOL,    lm_gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      LM_GGUF_TYPE_UINT8,   lm_gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     LM_GGUF_TYPE_UINT16,  lm_gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     LM_GGUF_TYPE_UINT32,  lm_gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     LM_GGUF_TYPE_UINT64,  lm_gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       LM_GGUF_TYPE_INT8,    lm_gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      LM_GGUF_TYPE_INT16,   lm_gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      LM_GGUF_TYPE_INT32,   lm_gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      LM_GGUF_TYPE_INT64,   lm_gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        LM_GGUF_TYPE_FLOAT32, lm_gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       LM_GGUF_TYPE_FLOAT64, lm_gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, LM_GGUF_TYPE_STRING,  lm_gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr lm_gguf_type gt = LM_GGUF_TYPE_STRING;

        static std::string getter(const lm_gguf_context * ctx, const int kid) {
            return lm_gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const lm_gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr lm_gguf_type gt = LM_GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const lm_gguf_context *ctx, const int k) {
            const enum lm_gguf_type arr_type = lm_gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                size_t(lm_gguf_get_arr_n(ctx, k)),
                arr_type == LM_GGUF_TYPE_STRING ? nullptr : lm_gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const lm_gguf_context * ctx, const int k) {
            const enum lm_gguf_type kt = lm_gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    lm_gguf_get_key(ctx, k), lm_gguf_type_name(kt), lm_gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const lm_gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const lm_gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, lm_gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const lm_gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(const std::string & key, T & result, bool required) {
        const int kid = lm_gguf_find_key(meta.get(), key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(enum llm_kv kid, T & result, bool required) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);

    template<typename T>
    bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
        const int kid = lm_gguf_find_key(meta.get(), key.c_str());

        if (kid < 0 || lm_gguf_get_kv_type(meta.get(), kid) != LM_GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

        switch (arr_info.gt) {
            case LM_GGUF_TYPE_UINT32:
            case LM_GGUF_TYPE_INT32:   LM_GGML_ASSERT((std::is_same<T,  int32_t>::value) ||
                                                (std::is_same<T, uint32_t>::value)); break;
            case LM_GGUF_TYPE_FLOAT32: LM_GGML_ASSERT((std::is_same<T,    float>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a float32/uint32/int32 array", key.c_str()));
        }

        result.resize(arr_info.length);
        result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);

        return true;
    }

    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
        const int kid = lm_gguf_find_key(meta.get(), key.c_str());

        if (kid < 0 || lm_gguf_get_kv_type(meta.get(), kid) != LM_GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

        switch (arr_info.gt) {
            case LM_GGUF_TYPE_UINT32:
            case LM_GGUF_TYPE_INT32:   LM_GGML_ASSERT((std::is_same<T,  int32_t>::value) ||
                                                (std::is_same<T, uint32_t>::value)); break;
            case LM_GGUF_TYPE_FLOAT32: LM_GGML_ASSERT((std::is_same<T,    float>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a float32/uint32/int32 array", key.c_str()));
        }

        if (arr_info.length > N_MAX) {
            throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
        }

        std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
        return get_arr(llm_kv(kid), result, required);
    }

    template<typename T>
    bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(meta.get(), key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
        return get_key(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
    template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
    template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
    template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

    template<>
    bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
        uint32_t tmp;
        const bool found = get_key(kid, tmp, required);
        if (found) {
            result = (enum llama_pooling_type) tmp;
        } else {
            result = LLAMA_POOLING_TYPE_UNSPECIFIED;
        }
        return found;
    }

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required) {
        const int kid = lm_gguf_find_key(meta.get(), key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        if (n > N_MAX) {
            throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", (uint32_t) n, (uint32_t) N_MAX, key.c_str()));
        }

        if (lm_gguf_get_kv_type(meta.get(), kid) == LM_GGUF_TYPE_ARRAY) {
            struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta.get(), kid);

            if (n != arr_info.length) {
                throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
            }

            return get_arr(key, result, required);
        }

        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    // TODO: this is not very clever - figure out something better
    template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);

llama_model_loader::llama_model_loader(
        const std::string & fname,
        std::vector<std::string> & splits,
        bool use_mmap,
        bool check_tensors,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    // Load the main GGUF
    struct lm_ggml_context * ctx = NULL;
    struct lm_gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta.reset(lm_gguf_init_from_file(fname.c_str(), params));
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));
    }

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    contexts.emplace_back(ctx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (lm_ggml_tensor * cur = lm_ggml_get_first_tensor(ctx); cur; cur = lm_ggml_get_next_tensor(ctx, cur)) {
        std::string tensor_name = std::string(cur->name);
        // make sure there is no duplicated tensor names
        if (weights_map.find(tensor_name) != weights_map.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", lm_ggml_get_name(cur)));
        }
        n_elements += lm_ggml_nelements(cur);
        n_bytes    += lm_ggml_nbytes(cur);
        weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, meta.get(), cur));
    }
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // Load additional GGML contexts
    if (n_split > 1) {
        // make sure the main file is loaded first
        uint16_t idx = 0;
        const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
        get_key(kv_split_no, idx);
        if (idx != 0) {
            throw std::runtime_error(format("illegal split file idx: %d (file: %s), model must be loaded with the first split", idx, fname.c_str()));
        }

        // generate list of splits if needed
        if (splits.empty()) {
            splits = llama_get_list_splits(fname, idx, n_split);
        }

        // in case user give a custom list of splits, check if it matches the expected number
        if (n_split != (uint16_t)splits.size()) {
            throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", splits.size(), n_split));
        }

        if (trace > 0) {
            LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
        }

        // load other splits
        for (idx = 1; idx < n_split; idx++) {
            const char * fname_split = splits[idx].c_str();

            struct lm_gguf_init_params split_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            lm_gguf_context_ptr ctx_gguf { lm_gguf_init_from_file(fname_split, split_params) };
            if (!ctx_gguf) {
                throw std::runtime_error(format("%s: failed to load GGUF split from %s", __func__, fname_split));
            }

            // check idx
            {
                const int kid = lm_gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
                if (kid < 0) {
                    throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
                }
                int idx_gguf = lm_gguf_get_val_u16(ctx_gguf.get(), kid);
                if (idx_gguf != idx) {
                    throw std::runtime_error(format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
                }
            }

            files.emplace_back(new llama_file(fname_split, "rb"));
            contexts.emplace_back(ctx);

            // Save tensors data offset info of the shard.
            for (lm_ggml_tensor * cur = lm_ggml_get_first_tensor(ctx); cur; cur = lm_ggml_get_next_tensor(ctx, cur)) {
                std::string tensor_name = std::string(cur->name);
                // make sure there is no duplicated tensor names
                if (weights_map.find(tensor_name) != weights_map.end()) {
                    throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", lm_ggml_get_name(cur)));
                }
                n_elements += lm_ggml_nelements(cur);
                n_bytes    += lm_ggml_nbytes(cur);
                weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
            }
        }

        get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

        // sanity check
        {
            const int n_tensors_loaded = (int) weights_map.size();
            if (n_tensors != n_tensors_loaded) {
                throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
            }
        }

        LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
    }

    n_kv      = lm_gguf_get_n_kv(meta.get());
    n_tensors = weights_map.size();

    fver = (enum llama_fver) lm_gguf_get_version(meta.get());

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum lm_ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum lm_ggml_type type_max = LM_GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const lm_ggml_tensor * tensor = w.tensor;

            enum lm_ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, lm_ggml_get_name(tensor), lm_ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        lm_ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case LM_GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case LM_GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case LM_GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case LM_GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case LM_GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case LM_GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case LM_GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case LM_GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case LM_GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case LM_GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case LM_GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case LM_GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case LM_GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case LM_GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case LM_GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case LM_GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case LM_GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case LM_GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case LM_GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case LM_GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case LM_GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case LM_GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case LM_GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case LM_GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, lm_ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = lm_gguf_get_key(meta.get(), i);
            const enum lm_gguf_type type   = lm_gguf_get_kv_type(meta.get(), i);
            const std::string type_name =
                type == LM_GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", lm_gguf_type_name(type), lm_gguf_type_name(lm_gguf_get_arr_type(meta.get(), i)), lm_gguf_get_arr_n(meta.get(), i))
                : lm_gguf_type_name(type);

            std::string value          = lm_gguf_kv_to_str(meta.get(), i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, lm_ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }

    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct lm_ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct lm_ggml_tensor * llama_model_loader::require_tensor_meta(const std::string & name) const {
    struct lm_ggml_tensor * tensor = get_tensor_meta(name.c_str());
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }
    return tensor;
}

const struct lm_ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const struct lm_ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < LM_GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

struct lm_ggml_tensor * llama_model_loader::create_tensor(struct lm_ggml_context * ctx, const std::string & name, const std::initializer_list<int64_t> & ne, int flags) {
    const struct lm_ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    bool duplicated = flags & TENSOR_DUPLICATED;

    struct lm_ggml_tensor * tensor = lm_ggml_dup_tensor(ctx, cur);
    lm_ggml_set_name(tensor, lm_ggml_get_name(cur));

    if (duplicated) {
        size_data += lm_ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;

}

struct lm_ggml_tensor * llama_model_loader::create_tensor_as_view(struct lm_ggml_context * ctx, struct lm_ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required) {
    const struct lm_ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), lm_ggml_type_name(base->type), lm_ggml_type_name(cur->type)));
    }

    std::array<int64_t, LM_GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < LM_GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne.begin()[i] : 1;
    }

    struct lm_ggml_tensor * tensor = lm_ggml_view_4d(ctx, base,
                                    dims[0], dims[1], dims[2], dims[3],
                                    cur->nb[1], cur->nb[2], cur->nb[3],
                                    offset);

    lm_ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors() const {
    if (n_created != n_tensors) {
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            bool is_numa = false;

            auto * dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                auto * reg = lm_ggml_backend_dev_backend_reg(dev);
                auto * is_numa_fn = (decltype(lm_ggml_is_numa) *) lm_ggml_backend_reg_get_proc_address(reg, "lm_ggml_backend_cpu_is_numa");
                if (is_numa_fn) {
                    is_numa = is_numa_fn();
                }
            }

            std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa);
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += lm_ggml_nbytes(it.second.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, lm_ggml_context * ctx) const {
    LM_GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (lm_ggml_tensor * tensor = lm_ggml_get_first_tensor(ctx); tensor; tensor = lm_ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = get_weight(lm_ggml_get_name(tensor));
        if (!weight || weight->idx != idx) {
            continue;
        }
        *first = std::min(*first, weight->offs);
        *last  = std::max(*last,  weight->offs + lm_ggml_nbytes(tensor));
    }
}

void llama_model_loader::load_data_for(struct lm_ggml_tensor * cur) const {
    const auto & w = require_weight(lm_ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, lm_ggml_nbytes(cur));
        }
    } else {
        LM_GGML_ASSERT(cur->data != nullptr);
        LM_GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, lm_ggml_nbytes(cur));
    }

    if (check_tensors && !lm_ggml_validate_row_data(cur->type, cur->data, lm_ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", lm_ggml_get_name(cur)));
    }
}

bool llama_model_loader::load_all_data(
        struct lm_ggml_context * ctx,
        llama_buf_map & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    LM_GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<lm_ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024; // 1MB

    std::vector<lm_ggml_backend_buffer_t> host_buffers;
    std::vector<lm_ggml_backend_event_t> events;
    std::vector<void *> host_ptrs;
    size_t buffer_idx = 0; // buffer to use for async loads
    lm_ggml_backend_t upload_backend = [&](const char * func) -> lm_ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = lm_ggml_backend_buffer_get_type(buf);
        auto * dev = lm_ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                lm_ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != lm_ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                lm_ggml_backend_buft_name(buft), lm_ggml_backend_dev_name(dev));
            return nullptr;
        }

        lm_ggml_backend_dev_props props;
        lm_ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                lm_ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = lm_ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                lm_ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = lm_ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    lm_ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(lm_ggml_backend_buffer_get_base(buf));

            auto * event = lm_ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    lm_ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }

        lm_ggml_backend_t backend = lm_ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                lm_ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
            lm_ggml_backend_dev_name(lm_ggml_backend_get_device(upload_backend)),
            lm_ggml_backend_buft_name(lm_ggml_backend_buffer_get_type(bufs.at(0))),
            lm_ggml_backend_name(upload_backend));
    }

    for (struct lm_ggml_tensor * cur = lm_ggml_get_first_tensor(ctx); cur != NULL; cur = lm_ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(lm_ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = lm_ggml_nbytes(cur);

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            lm_ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, lm_ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            LM_GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                lm_ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                lm_ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);
            if (lm_ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, lm_ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    file->seek(weight->offs, SEEK_SET);

                    size_t bytes_read = 0;

                    while (bytes_read < n_size) {
                        size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                        lm_ggml_backend_event_synchronize(events[buffer_idx]);
                        file->read_raw(host_ptrs[buffer_idx], read_iteration);
                        lm_ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                        lm_ggml_backend_event_record(events[buffer_idx], upload_backend);

                        bytes_read += read_iteration;
                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    read_buf.resize(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    lm_ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !lm_ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", lm_ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        lm_ggml_backend_event_synchronize(event);
        lm_ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        lm_ggml_backend_buffer_free(buf);
    }
    lm_ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, lm_ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

std::string llama_model_loader::ftype_name() const {
    return llama_model_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_model_ftype_name(ftype).c_str());
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}
