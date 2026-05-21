#include "cactus_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define private public
#include "engine.h"
#undef private
#include "cactus_kernels.h"
#include "stb_image_resize2.h"
#include "utils.h"
#include "wav.h"

namespace {

constexpr size_t kResponseBufferSize = 1 << 20;
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct Args {
    std::string command;
    std::string model;
    std::string prompt;
    std::string image;
    std::string audio;
    std::string messages_json;
    int max_tokens = 32;
};

std::string arg_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    if (argc < 2) throw std::runtime_error("usage: integration_runner <complete|transcribe|transcribe-pcm> [options]");
    Args args;
    args.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--model") args.model = arg_value(i, argc, argv);
        else if (flag == "--prompt") args.prompt = arg_value(i, argc, argv);
        else if (flag == "--image") args.image = arg_value(i, argc, argv);
        else if (flag == "--audio") args.audio = arg_value(i, argc, argv);
        else if (flag == "--messages-json") args.messages_json = arg_value(i, argc, argv);
        else if (flag == "--max-tokens") args.max_tokens = std::max(0, std::stoi(arg_value(i, argc, argv)));
        else throw std::runtime_error("unknown flag: " + flag);
    }
    if (args.model.empty()) throw std::runtime_error("--model is required");
    return args;
}

std::string escape_json(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << "\\u00";
                    const char* hex = "0123456789abcdef";
                    out << hex[(ch >> 4) & 0x0f] << hex[ch & 0x0f];
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

template <typename T>
std::string json_array(const std::vector<T>& values, size_t limit = std::numeric_limits<size_t>::max()) {
    std::ostringstream out;
    out << "[";
    const size_t n = std::min(values.size(), limit);
    for (size_t i = 0; i < n; ++i) {
        if (i) out << ",";
        out << values[i];
    }
    out << "]";
    return out.str();
}

std::string json_float_array(const std::vector<float>& values, size_t limit = std::numeric_limits<size_t>::max()) {
    std::ostringstream out;
    out << "[";
    const size_t n = std::min(values.size(), limit);
    for (size_t i = 0; i < n; ++i) {
        if (i) out << ",";
        if (std::isfinite(values[i])) out << values[i];
        else out << "null";
    }
    out << "]";
    return out.str();
}

uint64_t fnv1a_bytes(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = kFnvOffset;
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

template <typename T>
std::vector<T> first_values(const std::vector<T>& values, size_t n) {
    return std::vector<T>(values.begin(), values.begin() + std::min(values.size(), n));
}

std::vector<uint32_t> last_values(const std::vector<uint32_t>& values, size_t n) {
    if (values.size() <= n) return values;
    return std::vector<uint32_t>(values.end() - static_cast<std::ptrdiff_t>(n), values.end());
}

int run_e002_tokens(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for token probe: " + args.model);
    }
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) throw std::runtime_error("model has no tokenizer");

    cactus::engine::ChatMessage message;
    message.role = "user";
    message.content = args.prompt.empty() ? "Respond briefly." : args.prompt;
    message.images.push_back(args.image);
    std::vector<cactus::engine::ChatMessage> messages{message};

    const std::string full_prompt = tokenizer->format_chat_prompt(messages, true);
    const std::string no_generation_prompt = tokenizer->format_chat_prompt(messages, false);
    const std::vector<uint32_t> tokens = tokenizer->encode(full_prompt);
    const std::vector<uint32_t> prefix_tokens = tokenizer->encode(no_generation_prompt);
    const uint32_t image_token_id = model->get_config().image_token_id ? model->get_config().image_token_id : tokenizer->get_image_token_id();

    size_t image_start = tokens.size();
    size_t image_count = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == image_token_id) {
            if (image_count == 0) image_start = i;
            ++image_count;
        }
    }

    std::ostringstream out;
    out << "{"
        << "\"token_count\":" << tokens.size() << ","
        << "\"assistant_generation_start\":" << prefix_tokens.size() << ","
        << "\"image_token_id\":" << image_token_id << ","
        << "\"image_token_start\":" << (image_count ? static_cast<long long>(image_start) : -1) << ","
        << "\"image_token_count\":" << image_count << ","
        << "\"first_20\":" << json_array(first_values(tokens, 20)) << ","
        << "\"last_20\":" << json_array(last_values(tokens, 20)) << ","
        << "\"input_ids\":" << json_array(tokens) << ","
        << "\"prompt_prefix\":\"" << escape_json(full_prompt.substr(0, 320)) << "\""
        << "}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_preprocess(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    cactus::engine::Config config;
    if (!config.from_json(args.model + "/config.txt")) {
        throw std::runtime_error("failed to read config.txt for preprocess probe: " + args.model);
    }
    auto prep = cactus::engine::preprocess_gemma4_image(args.image, config);
    double sum = 0.0;
    float min_v = prep.pixel_values.empty() ? 0.0f : prep.pixel_values[0];
    float max_v = min_v;
    for (float v : prep.pixel_values) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<double>(v);
    }
    size_t valid_positions = 0;
    for (size_t i = 0; i + 1 < prep.pixel_position_ids.size(); i += 2) {
        if (prep.pixel_position_ids[i] >= 0 && prep.pixel_position_ids[i + 1] >= 0) ++valid_positions;
    }
    std::vector<int64_t> first_positions;
    const size_t position_rows = std::min<size_t>(16, prep.pixel_position_ids.size() / 2);
    first_positions.reserve(position_rows * 2);
    for (size_t i = 0; i < position_rows * 2; ++i) first_positions.push_back(prep.pixel_position_ids[i]);

    std::ostringstream out;
    out << std::setprecision(9)
        << "{"
        << "\"pixel_values_shape\":[1," << prep.max_patches << "," << prep.patch_dim << "],"
        << "\"pixel_position_ids_shape\":[1," << prep.max_patches << ",2],"
        << "\"num_patches\":" << prep.num_patches << ","
        << "\"max_patches\":" << prep.max_patches << ","
        << "\"patch_dim\":" << prep.patch_dim << ","
        << "\"pixel_values_min\":" << min_v << ","
        << "\"pixel_values_max\":" << max_v << ","
        << "\"pixel_values_mean\":" << (prep.pixel_values.empty() ? 0.0 : sum / static_cast<double>(prep.pixel_values.size())) << ","
        << "\"pixel_values_hash\":\"" << hex64(fnv1a_bytes(prep.pixel_values.data(), prep.pixel_values.size() * sizeof(float))) << "\","
        << "\"pixel_position_ids_hash\":\"" << hex64(fnv1a_bytes(prep.pixel_position_ids.data(), prep.pixel_position_ids.size() * sizeof(int64_t))) << "\","
        << "\"valid_position_count\":" << valid_positions << ","
        << "\"first_pixel_values\":" << json_array(first_values(prep.pixel_values, 32)) << ","
        << "\"first_position_rows_flat\":" << json_array(first_positions) << ","
        << "\"pixel_values\":" << json_array(prep.pixel_values) << ","
        << "\"pixel_position_ids\":" << json_array(prep.pixel_position_ids)
        << "}\n";
    std::cout << out.str();
    return 0;
}

struct E003Variant {
    std::string name;
    std::vector<float> resized;
    std::vector<float> pixel_values;
};

void resize_float_default(const unsigned char* raw, int width, int height, std::vector<float>& resized, int target_w, int target_h) {
    std::vector<float> src_float(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < src_float.size(); ++i) src_float[i] = static_cast<float>(raw[i]);
    if (!stbir_resize_float_linear(src_float.data(), width, height, 0, resized.data(), target_w, target_h, 0, STBIR_RGB)) {
        throw std::runtime_error("stbir_resize_float_linear failed");
    }
}

void resize_uint8_default(const unsigned char* raw, int width, int height, std::vector<float>& resized, int target_w, int target_h) {
    std::vector<unsigned char> tmp(static_cast<size_t>(target_w) * target_h * 3);
    if (!stbir_resize_uint8_linear(raw, width, height, 0, tmp.data(), target_w, target_h, 0, STBIR_RGB)) {
        throw std::runtime_error("stbir_resize_uint8_linear failed");
    }
    for (size_t i = 0; i < tmp.size(); ++i) resized[i] = static_cast<float>(tmp[i]);
}

void resize_float_triangle(const unsigned char* raw, int width, int height, std::vector<float>& resized, int target_w, int target_h) {
    std::vector<float> src_float(static_cast<size_t>(width) * height * 3);
    for (size_t i = 0; i < src_float.size(); ++i) src_float[i] = static_cast<float>(raw[i]);
    STBIR_RESIZE resize;
    stbir_resize_init(&resize, src_float.data(), width, height, 0, resized.data(), target_w, target_h, 0, STBIR_RGB, STBIR_TYPE_FLOAT);
    stbir_set_filters(&resize, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE);
    if (!stbir_resize_extended(&resize)) throw std::runtime_error("triangle float resize failed");
}

void resize_uint8_triangle(const unsigned char* raw, int width, int height, std::vector<float>& resized, int target_w, int target_h) {
    std::vector<unsigned char> tmp(static_cast<size_t>(target_w) * target_h * 3);
    STBIR_RESIZE resize;
    stbir_resize_init(&resize, raw, width, height, 0, tmp.data(), target_w, target_h, 0, STBIR_RGB, STBIR_TYPE_UINT8);
    stbir_set_filters(&resize, STBIR_FILTER_TRIANGLE, STBIR_FILTER_TRIANGLE);
    if (!stbir_resize_extended(&resize)) throw std::runtime_error("triangle uint8 resize failed");
    for (size_t i = 0; i < tmp.size(); ++i) resized[i] = static_cast<float>(tmp[i]);
}

std::vector<float> pack_gemma4_patches(const std::vector<float>& resized, int target_w, int target_h, int patch_size,
                                       size_t max_patches, size_t patch_dim, float rescale_factor) {
    const int patch_h = target_h / patch_size;
    const int patch_w = target_w / patch_size;
    std::vector<float> pixel_values(max_patches * patch_dim, 0.0f);
    for (int py = 0; py < patch_h; ++py) {
        for (int px = 0; px < patch_w; ++px) {
            const size_t patch_idx = static_cast<size_t>(py) * patch_w + px;
            float* dst = pixel_values.data() + patch_idx * patch_dim;
            for (int y = 0; y < patch_size; ++y) {
                const int img_y = py * patch_size + y;
                for (int x = 0; x < patch_size; ++x) {
                    const int img_x = px * patch_size + x;
                    const size_t src_off = (static_cast<size_t>(img_y) * target_w + img_x) * 3;
                    const size_t dst_off = (static_cast<size_t>(y) * patch_size + x) * 3;
                    dst[dst_off + 0] = resized[src_off + 0] * rescale_factor;
                    dst[dst_off + 1] = resized[src_off + 1] * rescale_factor;
                    dst[dst_off + 2] = resized[src_off + 2] * rescale_factor;
                }
            }
        }
    }
    return pixel_values;
}

void write_variant_json(std::ostream& out, const E003Variant& variant, bool comma) {
    double sum = 0.0;
    float min_v = variant.pixel_values.empty() ? 0.0f : variant.pixel_values[0];
    float max_v = min_v;
    for (float v : variant.pixel_values) {
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<double>(v);
    }
    float resized_min = variant.resized.empty() ? 0.0f : variant.resized[0];
    float resized_max = resized_min;
    for (float v : variant.resized) {
        resized_min = std::min(resized_min, v);
        resized_max = std::max(resized_max, v);
    }
    if (comma) out << ",";
    out << "\"" << variant.name << "\":{"
        << "\"resized_min\":" << resized_min << ","
        << "\"resized_max\":" << resized_max << ","
        << "\"first_resized_values\":" << json_array(first_values(variant.resized, 32)) << ","
        << "\"pixel_values_min\":" << min_v << ","
        << "\"pixel_values_max\":" << max_v << ","
        << "\"pixel_values_mean\":" << (variant.pixel_values.empty() ? 0.0 : sum / static_cast<double>(variant.pixel_values.size())) << ","
        << "\"first_pixel_values\":" << json_array(first_values(variant.pixel_values, 32)) << ","
        << "\"pixel_values\":" << json_array(variant.pixel_values)
        << "}";
}

std::string precision_name(Precision precision) {
    switch (precision) {
        case Precision::FP32: return "FP32";
        case Precision::FP16: return "FP16";
        case Precision::INT8: return "INT8";
        default: return "UNKNOWN";
    }
}

std::string op_type_name(OpType op) {
    switch (op) {
        case OpType::INPUT: return "INPUT";
        case OpType::PRECISION_CAST: return "PRECISION_CAST";
        case OpType::ADD: return "ADD";
        case OpType::ADD_CLIPPED: return "ADD_CLIPPED";
        case OpType::SUBTRACT: return "SUBTRACT";
        case OpType::MULTIPLY: return "MULTIPLY";
        case OpType::DIVIDE: return "DIVIDE";
        case OpType::ABS: return "ABS";
        case OpType::POW: return "POW";
        case OpType::FLATTEN: return "FLATTEN";
        case OpType::VIEW: return "VIEW";
        case OpType::MATMUL: return "MATMUL";
        case OpType::TRANSPOSE: return "TRANSPOSE";
        case OpType::RESHAPE: return "RESHAPE";
        case OpType::SLICE: return "SLICE";
        case OpType::GATHER: return "GATHER";
        case OpType::EMBEDDING: return "EMBEDDING";
        case OpType::BILINEAR_INTERPOLATION: return "BILINEAR_INTERPOLATION";
        case OpType::SUM: return "SUM";
        case OpType::MEAN: return "MEAN";
        case OpType::VARIANCE: return "VARIANCE";
        case OpType::MIN: return "MIN";
        case OpType::MAX: return "MAX";
        case OpType::CUMSUM: return "CUMSUM";
        case OpType::RMS_NORM: return "RMS_NORM";
        case OpType::ROPE: return "ROPE";
        case OpType::ROPE_GPTJ: return "ROPE_GPTJ";
        case OpType::SOFTMAX: return "SOFTMAX";
        case OpType::ATTENTION: return "ATTENTION";
        case OpType::ATTENTION_INT8_HYBRID: return "ATTENTION_INT8_HYBRID";
        case OpType::REL_POS_BIAS: return "REL_POS_BIAS";
        case OpType::CONV1D_CAUSAL: return "CONV1D_CAUSAL";
        case OpType::CONV1D_K3: return "CONV1D_K3";
        case OpType::CONV1D_K7S3: return "CONV1D_K7S3";
        case OpType::CONV1D: return "CONV1D";
        case OpType::CONV1D_SAME_DEPTHWISE_K9: return "CONV1D_SAME_DEPTHWISE_K9";
        case OpType::CONV1D_POINTWISE: return "CONV1D_POINTWISE";
        case OpType::CONV2D_K3S2P1: return "CONV2D_K3S2P1";
        case OpType::CONV2D_DEPTHWISE_K3S2P1: return "CONV2D_DEPTHWISE_K3S2P1";
        case OpType::CONV2D_POINTWISE_1X1: return "CONV2D_POINTWISE_1X1";
        case OpType::GLU: return "GLU";
        case OpType::BATCHNORM: return "BATCHNORM";
        case OpType::SCALAR_ADD: return "SCALAR_ADD";
        case OpType::SCALAR_SUBTRACT: return "SCALAR_SUBTRACT";
        case OpType::SCALAR_MULTIPLY: return "SCALAR_MULTIPLY";
        case OpType::SCALAR_DIVIDE: return "SCALAR_DIVIDE";
        case OpType::SCALAR_EXP: return "SCALAR_EXP";
        case OpType::SCALAR_SQRT: return "SCALAR_SQRT";
        case OpType::SCALAR_COS: return "SCALAR_COS";
        case OpType::SCALAR_SIN: return "SCALAR_SIN";
        case OpType::SCALAR_LOG: return "SCALAR_LOG";
        case OpType::RELU: return "RELU";
        case OpType::SILU: return "SILU";
        case OpType::GELU: return "GELU";
        case OpType::GELU_ERF: return "GELU_ERF";
        case OpType::SIGMOID: return "SIGMOID";
        case OpType::TANH: return "TANH";
        case OpType::SAMPLE: return "SAMPLE";
        case OpType::CONCAT: return "CONCAT";
        case OpType::CAT: return "CAT";
        case OpType::SCATTER_TOPK: return "SCATTER_TOPK";
        case OpType::TOPK: return "TOPK";
        case OpType::LAYERNORM: return "LAYERNORM";
        case OpType::GROUPNORM: return "GROUPNORM";
        case OpType::MOE_LAYER: return "MOE_LAYER";
        case OpType::INDEX: return "INDEX";
        case OpType::PERSISTENT: return "PERSISTENT";
        case OpType::LSTM_CELL: return "LSTM_CELL";
        case OpType::GATED_DELTANET_DECODE: return "GATED_DELTANET_DECODE";
        case OpType::GATED_DELTANET_PREFILL: return "GATED_DELTANET_PREFILL";
        case OpType::STFT: return "STFT";
        case OpType::ALTUP_PREDICT: return "ALTUP_PREDICT";
        case OpType::ALTUP_CORRECT: return "ALTUP_CORRECT";
        case OpType::GAUSSIAN_TOPK: return "GAUSSIAN_TOPK";
        case OpType::MAXPOOL1D: return "MAXPOOL1D";
        case OpType::BILSTM_SEQUENCE: return "BILSTM_SEQUENCE";
        case OpType::LEAKY_RELU: return "LEAKY_RELU";
        case OpType::CONV2D_K3S1P1: return "CONV2D_K3S1P1";
        case OpType::STATS_POOL: return "STATS_POOL";
        case OpType::WEIGHTED_STATS_POOL: return "WEIGHTED_STATS_POOL";
        case OpType::KV_CACHE_STATE: return "KV_CACHE_STATE";
        case OpType::KV_CACHE_APPEND: return "KV_CACHE_APPEND";
        case OpType::ATTENTION_CACHED: return "ATTENTION_CACHED";
        case OpType::CONV_CACHE_STATE: return "CONV_CACHE_STATE";
        case OpType::CONV_CACHE_APPEND: return "CONV_CACHE_APPEND";
        case OpType::RFFT: return "RFFT";
        case OpType::IRFFT: return "IRFFT";
        case OpType::MEL_FILTER_BANK: return "MEL_FILTER_BANK";
        case OpType::SPECTROGRAM: return "SPECTROGRAM";
        case OpType::IMAGE_PREPROCESS: return "IMAGE_PREPROCESS";
        case OpType::CLAMP: return "CLAMP";
        case OpType::DENSE_MLP_TQ_FUSED: return "DENSE_MLP_TQ_FUSED";
        case OpType::NOT_EQUAL: return "NOT_EQUAL";
        case OpType::SCALAR_NOT_EQUAL: return "SCALAR_NOT_EQUAL";
    }
    return "UNKNOWN";
}

std::vector<float> bytes_to_floats(const std::vector<uint8_t>& bytes, Precision precision) {
    std::vector<float> values;
    if (precision == Precision::FP32) {
        const size_t count = bytes.size() / sizeof(float);
        values.resize(count);
        const auto* src = reinterpret_cast<const float*>(bytes.data());
        for (size_t i = 0; i < count; ++i) values[i] = src[i];
    } else if (precision == Precision::FP16) {
        const size_t count = bytes.size() / sizeof(__fp16);
        values.resize(count);
        const auto* src = reinterpret_cast<const __fp16*>(bytes.data());
        for (size_t i = 0; i < count; ++i) values[i] = static_cast<float>(src[i]);
    } else if (precision == Precision::INT8) {
        const size_t count = bytes.size();
        values.resize(count);
        const auto* src = reinterpret_cast<const int8_t*>(bytes.data());
        for (size_t i = 0; i < count; ++i) values[i] = static_cast<float>(src[i]);
    }
    return values;
}

std::vector<float> buffer_to_floats(const BufferDesc& desc) {
    std::vector<float> values;
    const void* data = desc.get_data();
    if (!data) return values;
    if (desc.precision == Precision::FP32) {
        values.resize(desc.byte_size / sizeof(float));
        const auto* src = static_cast<const float*>(data);
        for (size_t i = 0; i < values.size(); ++i) values[i] = src[i];
    } else if (desc.precision == Precision::FP16) {
        values.resize(desc.byte_size / sizeof(__fp16));
        const auto* src = static_cast<const __fp16*>(data);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(src[i]);
    } else if (desc.precision == Precision::INT8) {
        values.resize(desc.byte_size);
        const auto* src = static_cast<const int8_t*>(data);
        for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(src[i]);
    }
    return values;
}

void write_float_feature_json(std::ostream& out,
                              const std::string& name,
                              const std::vector<float>& values,
                              const std::vector<size_t>& shape,
                              Precision precision,
                              bool comma) {
    double sum = 0.0;
    float min_v = values.empty() ? 0.0f : values[0];
    float max_v = min_v;
    double l2 = 0.0;
    size_t finite_count = 0;
    size_t nonfinite_count = 0;
    for (float v : values) {
        if (!std::isfinite(v)) {
            ++nonfinite_count;
            continue;
        }
        if (finite_count == 0) {
            min_v = v;
            max_v = v;
        }
        ++finite_count;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<double>(v);
        l2 += static_cast<double>(v) * static_cast<double>(v);
    }
    if (comma) out << ",";
    out << "\"" << escape_json(name) << "\":{"
        << "\"shape\":" << json_array(shape) << ","
        << "\"precision\":\"" << precision_name(precision) << "\","
        << "\"count\":" << values.size() << ","
        << "\"finite_count\":" << finite_count << ","
        << "\"nonfinite_count\":" << nonfinite_count << ","
        << "\"min\":" << min_v << ","
        << "\"max\":" << max_v << ","
        << "\"mean\":" << (finite_count == 0 ? 0.0 : sum / static_cast<double>(finite_count)) << ","
        << "\"l2\":" << std::sqrt(l2) << ","
        << "\"first_values\":" << json_float_array(values, 32) << ","
        << "\"values\":" << json_float_array(values)
        << "}";
}

int run_e003_preprocess_variants(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    cactus::engine::Config config;
    if (!config.from_json(args.model + "/config.txt")) {
        throw std::runtime_error("failed to read config.txt for E003 probe: " + args.model);
    }

    const int patch_size = static_cast<int>(config.vision_patch_size ? config.vision_patch_size : 16);
    const int pooling_k = static_cast<int>(config.vision_pooling_kernel_size ? config.vision_pooling_kernel_size : 3);
    const size_t max_patches = static_cast<size_t>(config.vision_default_output_length ? config.vision_default_output_length : 280) * pooling_k * pooling_k;
    const int side_multiple = pooling_k * patch_size;
    const size_t patch_dim = static_cast<size_t>(3) * patch_size * patch_size;
    const float rescale_factor = config.rescale_factor > 0.0f ? config.rescale_factor : (1.0f / 255.0f);

    int width = 0, height = 0, channels = 0;
    unsigned char* raw = cactus_image_load(args.image.c_str(), &width, &height, &channels, 3);
    if (!raw) throw std::runtime_error("failed to load image for E003 probe: " + args.image);

    const double target_pixels = static_cast<double>(max_patches) * patch_size * patch_size;
    const double pixel_count = std::max(1.0, static_cast<double>(width) * static_cast<double>(height));
    const double factor = std::sqrt(target_pixels / pixel_count);
    int target_h = static_cast<int>(std::floor(factor * height / side_multiple)) * side_multiple;
    int target_w = static_cast<int>(std::floor(factor * width / side_multiple)) * side_multiple;
    if (target_h == 0) target_h = side_multiple;
    if (target_w == 0) target_w = side_multiple;

    std::vector<E003Variant> variants;
    for (const std::string& name : {"stb_default_float", "stb_default_uint8", "stb_triangle_float", "stb_triangle_uint8"}) {
        E003Variant variant;
        variant.name = name;
        variant.resized.assign(static_cast<size_t>(target_w) * target_h * 3, 0.0f);
        if (name == "stb_default_float") resize_float_default(raw, width, height, variant.resized, target_w, target_h);
        else if (name == "stb_default_uint8") resize_uint8_default(raw, width, height, variant.resized, target_w, target_h);
        else if (name == "stb_triangle_float") resize_float_triangle(raw, width, height, variant.resized, target_w, target_h);
        else resize_uint8_triangle(raw, width, height, variant.resized, target_w, target_h);
        variant.pixel_values = pack_gemma4_patches(variant.resized, target_w, target_h, patch_size, max_patches, patch_dim, rescale_factor);
        variants.push_back(std::move(variant));
    }
    cactus_image_free(raw);

    std::ostringstream out;
    out << std::setprecision(9)
        << "{"
        << "\"source_width\":" << width << ","
        << "\"source_height\":" << height << ","
        << "\"target_width\":" << target_w << ","
        << "\"target_height\":" << target_h << ","
        << "\"patch_size\":" << patch_size << ","
        << "\"pooling_kernel_size\":" << pooling_k << ","
        << "\"max_patches\":" << max_patches << ","
        << "\"patch_dim\":" << patch_dim << ","
        << "\"rescale_factor\":" << rescale_factor << ","
        << "\"variants\":{";
    for (size_t i = 0; i < variants.size(); ++i) {
        write_variant_json(out, variants[i], i > 0);
    }
    out << "}}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_vision(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for vision probe: " + args.model);
    }
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) throw std::runtime_error("model has no tokenizer");
    cactus::engine::ChatMessage message;
    message.role = "user";
    message.content = args.prompt.empty()
        ? "What animal is in this image? Reply in one complete short sentence."
        : args.prompt;
    message.images.push_back(args.image);
    std::vector<cactus::engine::ChatMessage> messages{message};
    std::vector<uint32_t> tokens = tokenizer->encode(tokenizer->format_chat_prompt(messages, true));
    if (!model->run_chunk_prefill_path(tokens, {args.image}, {})) {
        throw std::runtime_error("chunk prefill path failed for vision probe");
    }

    std::ostringstream out;
    out << std::setprecision(9) << "{\"inputs\":{";
    if (model->vision_encoder_) {
        for (size_t i = 0; i < model->vision_encoder_->logical_inputs.size(); ++i) {
            if (i) out << ",";
            const std::string& input_name = model->vision_encoder_->logical_inputs[i];
            const size_t node_id = static_cast<size_t>(model->vision_encoder_->runtime_input_node_ids[i]);
            const auto& desc = model->vision_encoder_->graph->get_output_buffer(node_id);
            out << "\"" << escape_json(input_name) << "\":{"
                << "\"node_id\":" << node_id << ","
                << "\"shape\":" << json_array(desc.shape) << ","
                << "\"precision\":\"" << precision_name(desc.precision) << "\","
                << "\"byte_size\":" << desc.byte_size
                << "}";
        }
    }
    out << "},\"features\":{";
    bool first = true;
    for (const auto& kv : model->media_features_) {
        const std::string& name = kv.first;
        Precision precision = model->media_feature_precisions_[name];
        std::vector<float> values = bytes_to_floats(kv.second, precision);
        write_float_feature_json(out, name, values, model->media_feature_shapes_[name], precision, !first);
        first = false;
    }
    out << "}}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_vision_node_summary(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for vision node summary: " + args.model);
    }
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) throw std::runtime_error("model has no tokenizer");
    cactus::engine::ChatMessage message;
    message.role = "user";
    message.content = args.prompt.empty()
        ? "What animal is in this image? Reply in one complete short sentence."
        : args.prompt;
    message.images.push_back(args.image);
    std::vector<cactus::engine::ChatMessage> messages{message};
    std::vector<uint32_t> tokens = tokenizer->encode(tokenizer->format_chat_prompt(messages, true));
    if (!model->run_chunk_prefill_path(tokens, {args.image}, {})) {
        throw std::runtime_error("chunk prefill path failed for vision node summary");
    }
    if (!model->vision_encoder_ || !model->vision_encoder_->graph) {
        throw std::runtime_error("model has no vision encoder graph");
    }

    const auto& graph = *model->vision_encoder_->graph;
    std::ostringstream out;
    out << std::setprecision(9) << "{\"nodes\":[";
    bool first = true;
    for (const auto& node_ptr : graph.nodes_) {
        const GraphNode& node = *node_ptr;
        const auto& desc = node.output_buffer;
        const bool interesting =
            node.id < 25 ||
            (node.id >= 660 && node.id <= 710) ||
            desc.shape == std::vector<size_t>{1, 2520} ||
            desc.shape == std::vector<size_t>{1, 2520, 1} ||
            desc.shape == std::vector<size_t>{1, 2520, 2} ||
            desc.shape == std::vector<size_t>{1, 2520, 768} ||
            desc.shape == std::vector<size_t>{1, 256, 768} ||
            desc.shape == std::vector<size_t>{256, 768} ||
            desc.shape == std::vector<size_t>{256, 1536};
        if (!interesting) continue;
        std::vector<float> values = buffer_to_floats(desc);
        double sum = 0.0;
        float min_v = values.empty() ? 0.0f : values[0];
        float max_v = min_v;
        size_t finite_count = 0;
        size_t nonfinite_count = 0;
        for (float v : values) {
            if (!std::isfinite(v)) {
                ++nonfinite_count;
                continue;
            }
            if (finite_count == 0) {
                min_v = v;
                max_v = v;
            }
            ++finite_count;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum += static_cast<double>(v);
        }
        if (!first) out << ",";
        first = false;
        out << "{"
            << "\"node_id\":" << node.id << ","
            << "\"op\":\"" << op_type_name(node.op_type) << "\","
            << "\"inputs\":" << json_array(node.input_ids) << ","
            << "\"shape\":" << json_array(desc.shape) << ","
            << "\"precision\":\"" << precision_name(desc.precision) << "\","
            << "\"byte_size\":" << desc.byte_size << ","
            << "\"value_count\":" << values.size() << ","
            << "\"finite_count\":" << finite_count << ","
            << "\"nonfinite_count\":" << nonfinite_count << ","
            << "\"min\":" << min_v << ","
            << "\"max\":" << max_v << ","
            << "\"mean\":" << (finite_count == 0 ? 0.0 : sum / static_cast<double>(finite_count)) << ","
            << "\"first_values\":" << json_float_array(values, 32) << ","
            << "\"values\":" << json_float_array(values)
            << "}";
    }
    out << "]}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_vision_selected_nodes(const Args& args) {
    if (args.image.empty()) throw std::runtime_error("--image is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for selected vision nodes: " + args.model);
    }
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) throw std::runtime_error("model has no tokenizer");
    cactus::engine::ChatMessage message;
    message.role = "user";
    message.content = args.prompt.empty()
        ? "What animal is in this image? Reply in one complete short sentence."
        : args.prompt;
    message.images.push_back(args.image);
    std::vector<cactus::engine::ChatMessage> messages{message};
    std::vector<uint32_t> tokens = tokenizer->encode(tokenizer->format_chat_prompt(messages, true));
    if (!model->run_chunk_prefill_path(tokens, {args.image}, {})) {
        throw std::runtime_error("chunk prefill path failed for selected vision nodes");
    }
    if (!model->vision_encoder_ || !model->vision_encoder_->graph) {
        throw std::runtime_error("model has no vision encoder graph");
    }

    const std::vector<size_t> selected = {
        672, 675, 699, 700,
        752, 753, 756, 757, 789, 792, 793, 825, 828, 829, 846, 847, 874, 875,
        2757, 2758, 2804, 2805, 2806, 2807, 2812, 2813, 2814, 2815, 2816,
    };
    const auto& graph = *model->vision_encoder_->graph;
    std::ostringstream out;
    out << std::setprecision(9) << "{\"nodes\":{";
    bool first = true;
    for (size_t node_id : selected) {
        auto it = graph.node_index_map_.find(node_id);
        if (it == graph.node_index_map_.end()) continue;
        const GraphNode& node = *graph.nodes_[it->second];
        const auto& desc = node.output_buffer;
        std::vector<float> values = buffer_to_floats(desc);
        if (!first) out << ",";
        first = false;
        out << "\"" << node_id << "\":{"
            << "\"node_id\":" << node.id << ","
            << "\"op\":\"" << op_type_name(node.op_type) << "\","
            << "\"inputs\":" << json_array(node.input_ids) << ","
            << "\"shape\":" << json_array(desc.shape) << ","
            << "\"precision\":\"" << precision_name(desc.precision) << "\","
            << "\"values\":" << json_float_array(values)
            << "}";
    }
    out << "}}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_audio_summary(const Args& args) {
    if (args.audio.empty()) throw std::runtime_error("--audio is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for audio summary: " + args.model);
    }
    AudioFP32 wav = load_wav(args.audio);
    std::vector<float> waveform_16k = resample_to_16k_fp32(wav.samples, wav.sample_rate);
    auto prep = cactus::audio::preprocess_audio_for_gemma4(std::move(waveform_16k), model->get_config());
    model->run_audio_encoder(prep.features);

    std::ostringstream out;
    out << std::setprecision(9) << "{"
        << "\"num_frames\":" << prep.num_frames << ","
        << "\"num_soft_tokens\":" << prep.num_soft_tokens << ","
        << "\"feature_count\":" << prep.features.size() << ","
        << "\"feature_first_values\":" << json_float_array(prep.features, 32) << ","
        << "\"media_features\":{";
    bool first_feature = true;
    for (const auto& kv : model->media_features_) {
        const std::string& name = kv.first;
        Precision precision = model->media_feature_precisions_[name];
        std::vector<float> values = bytes_to_floats(kv.second, precision);
        double sum = 0.0;
        float min_v = values.empty() ? 0.0f : values[0];
        float max_v = min_v;
        size_t finite_count = 0;
        size_t nonfinite_count = 0;
        for (float v : values) {
            if (!std::isfinite(v)) {
                ++nonfinite_count;
                continue;
            }
            if (finite_count == 0) {
                min_v = v;
                max_v = v;
            }
            ++finite_count;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum += static_cast<double>(v);
        }
        if (!first_feature) out << ",";
        first_feature = false;
        out << "\"" << name << "\":{"
            << "\"shape\":" << json_array(model->media_feature_shapes_[name]) << ","
            << "\"precision\":\"" << precision_name(precision) << "\","
            << "\"value_count\":" << values.size() << ","
            << "\"finite_count\":" << finite_count << ","
            << "\"nonfinite_count\":" << nonfinite_count << ","
            << "\"min\":" << min_v << ","
            << "\"max\":" << max_v << ","
            << "\"mean\":" << (finite_count == 0 ? 0.0 : sum / static_cast<double>(finite_count)) << ","
            << "\"first_values\":" << json_float_array(values, 32) << ","
            << "\"values\":" << json_float_array(values)
            << "}";
    }
    out << "}}\n";
    std::cout << out.str();
    return 0;
}

int run_e002_audio_node_summary(const Args& args) {
    if (args.audio.empty()) throw std::runtime_error("--audio is required");
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for audio node summary: " + args.model);
    }
    AudioFP32 wav = load_wav(args.audio);
    std::vector<float> waveform_16k = resample_to_16k_fp32(wav.samples, wav.sample_rate);
    auto prep = cactus::audio::preprocess_audio_for_gemma4(std::move(waveform_16k), model->get_config());
    model->run_audio_encoder(prep.features);
    if (!model->audio_encoder_ || !model->audio_encoder_->graph) {
        throw std::runtime_error("model has no audio encoder graph");
    }

    const auto& graph = *model->audio_encoder_->graph;
    std::ostringstream out;
    out << std::setprecision(9) << "{\"nodes\":[";
    bool first = true;
    for (size_t order = 0; order < graph.nodes_.size(); ++order) {
        const GraphNode& node = *graph.nodes_[order];
        const auto& desc = node.output_buffer;
        const auto& shape = desc.shape;
        bool interesting = node.op_type != OpType::INPUT;
        if (shape.empty()) interesting = false;
        if (!interesting) continue;

        std::vector<float> values = buffer_to_floats(desc);
        double sum = 0.0;
        float min_v = values.empty() ? 0.0f : values[0];
        float max_v = min_v;
        size_t finite_count = 0;
        size_t nonfinite_count = 0;
        for (float v : values) {
            if (!std::isfinite(v)) {
                ++nonfinite_count;
                continue;
            }
            if (finite_count == 0) {
                min_v = v;
                max_v = v;
            }
            ++finite_count;
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum += static_cast<double>(v);
        }
        if (!first) out << ",";
        first = false;
        out << "{"
            << "\"order\":" << order << ","
            << "\"node_id\":" << node.id << ","
            << "\"op\":\"" << op_type_name(node.op_type) << "\","
            << "\"inputs\":" << json_array(node.input_ids) << ","
            << "\"shape\":" << json_array(shape) << ","
            << "\"precision\":\"" << precision_name(desc.precision) << "\","
            << "\"value_count\":" << values.size() << ","
            << "\"finite_count\":" << finite_count << ","
            << "\"nonfinite_count\":" << nonfinite_count << ","
            << "\"min\":" << min_v << ","
            << "\"max\":" << max_v << ","
            << "\"mean\":" << (finite_count == 0 ? 0.0 : sum / static_cast<double>(finite_count)) << ","
            << "\"first_values\":" << json_float_array(values, 16)
            << "}";
    }
    out << "]}\n";
    std::cout << out.str();
    return 0;
}

int run_gemma_first_token(const Args& args) {
    auto model = cactus::engine::create_model(args.model);
    if (!model || !model->init(args.model, 4096, "", false)) {
        throw std::runtime_error("failed to initialize model for first token probe: " + args.model);
    }
    if (args.command == "gemma-first-token-media-step") {
        model->lm_encoder_ = nullptr;
    }
    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) throw std::runtime_error("model has no tokenizer");

    cactus::engine::ChatMessage message;
    message.role = "user";
    message.content = args.prompt.empty() ? "Respond briefly." : args.prompt;
    if (!args.image.empty()) message.images.push_back(args.image);

    std::vector<float> audio_features;
    size_t audio_num_frames = 0;
    if (!args.audio.empty()) {
        message.audio.push_back(args.audio);
        AudioFP32 wav = load_wav(args.audio);
        std::vector<float> waveform_16k = resample_to_16k_fp32(wav.samples, wav.sample_rate);
        auto prep = cactus::audio::preprocess_audio_for_gemma4(std::move(waveform_16k), model->get_config());
        audio_features = std::move(prep.features);
        audio_num_frames = prep.num_frames;
        message.audio_soft_token_count = prep.num_soft_tokens;
    }

    std::vector<cactus::engine::ChatMessage> messages{message};
    std::string prompt = tokenizer->format_chat_prompt(messages, true);
    std::vector<uint32_t> tokens = tokenizer->encode(prompt);
    if (tokens.empty()) throw std::runtime_error("first token probe produced empty prompt");
    std::vector<uint32_t> prefill_tokens(tokens.begin(), tokens.end() - 1);
    if (!prefill_tokens.empty()) {
        std::vector<std::vector<float>> audio_messages;
        if (!audio_features.empty()) audio_messages.push_back(std::move(audio_features));
        model->prefill_with_media(prefill_tokens, args.image.empty() ? std::vector<std::string>{} : std::vector<std::string>{args.image}, audio_messages);
    }
    uint32_t first = model->decode({tokens.back()}, 0.0f, 1.0f, 1);
    std::string text = tokenizer->decode({first});
    std::vector<std::pair<size_t, float>> top_logits;
    if (model->decoder_ && !model->decoder_->output_node_ids.empty()) {
        size_t out_node = static_cast<size_t>(model->decoder_->output_node_ids[0]);
        const auto& desc = model->decoder_->graph->get_output_buffer(out_node);
        std::vector<float> logits = buffer_to_floats(desc);
        if (!desc.shape.empty() && !logits.empty()) {
            size_t vocab = desc.shape.back();
            size_t seq = desc.shape.size() >= 2 ? desc.shape[desc.shape.size() - 2] : 1;
            size_t row = seq > 0 ? seq - 1 : 0;
            size_t start = row * vocab;
            size_t end = std::min(start + vocab, logits.size());
            for (size_t i = start; i < end; ++i) top_logits.emplace_back(i - start, logits[i]);
            size_t keep = std::min<size_t>(10, top_logits.size());
            std::partial_sort(
                top_logits.begin(),
                top_logits.begin() + keep,
                top_logits.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            top_logits.resize(keep);
        }
    }

    const uint32_t image_token_id = model->get_config().image_token_id ? model->get_config().image_token_id : tokenizer->get_image_token_id();
    const uint32_t audio_token_id = model->get_config().audio_token_id;
    size_t image_token_count = 0;
    size_t audio_token_count = 0;
    for (uint32_t token : tokens) {
        if (image_token_id != 0 && token == image_token_id) ++image_token_count;
        if (audio_token_id != 0 && token == audio_token_id) ++audio_token_count;
    }

    std::ostringstream out;
    out << "{"
        << "\"prompt_token_count\":" << tokens.size() << ","
        << "\"prefill_token_count\":" << prefill_tokens.size() << ","
        << "\"last_prompt_token\":" << tokens.back() << ","
        << "\"first_token\":" << first << ","
        << "\"first_text\":\"" << escape_json(text) << "\","
        << "\"eos_token\":" << tokenizer->get_eos_token() << ","
        << "\"is_eos\":" << (first == tokenizer->get_eos_token() ? "true" : "false") << ","
        << "\"image_token_count\":" << image_token_count << ","
        << "\"audio_token_id\":" << audio_token_id << ","
        << "\"audio_token_count\":" << audio_token_count << ","
        << "\"audio_num_frames\":" << audio_num_frames << ","
        << "\"top_logits\":[";
    for (size_t i = 0; i < top_logits.size(); ++i) {
        if (i) out << ",";
        out << "{\"token\":" << top_logits[i].first
            << ",\"score\":" << top_logits[i].second
            << ",\"text\":\"" << escape_json(tokenizer->decode({static_cast<uint32_t>(top_logits[i].first)})) << "\"}";
    }
    out << "],"
        << "\"prompt_prefix\":\"" << escape_json(prompt.substr(0, 320)) << "\""
        << "}\n";
    std::cout << out.str();
    return 0;
}

std::string build_messages(const Args& args) {
    if (!args.messages_json.empty()) return args.messages_json;
    std::ostringstream out;
    out << "[{\"role\":\"user\",\"content\":\"" << escape_json(args.prompt.empty() ? "Respond briefly." : args.prompt) << "\"";
    if (!args.image.empty()) out << ",\"images\":[\"" << escape_json(args.image) << "\"]";
    if (!args.audio.empty()) out << ",\"audio\":[\"" << escape_json(args.audio) << "\"]";
    out << "}]";
    return out.str();
}

std::string complete_options(int max_tokens) {
    std::ostringstream out;
    out << "{"
        << "\"max_tokens\":" << max_tokens << ","
        << "\"temperature\":0.0,"
        << "\"top_p\":1.0,"
        << "\"top_k\":1,"
        << "\"telemetry_enabled\":false,"
        << "\"auto_handoff\":false,"
        << "\"confidence_threshold\":0.0,"
        << "\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]"
        << "}";
    return out.str();
}

std::vector<uint8_t> read_wav_pcm16(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open wav: " + path);
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    if (size <= 44) throw std::runtime_error("wav file too small: " + path);
    in.seekg(44, std::ios::beg);
    std::vector<uint8_t> pcm(static_cast<size_t>(size - 44));
    in.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(pcm.size()));
    if (!in) throw std::runtime_error("failed to read wav pcm: " + path);
    return pcm;
}

int run_complete(const Args& args) {
    cactus_model_t model = cactus_init(args.model.c_str(), nullptr, false);
    if (!model) throw std::runtime_error("cactus_init failed for " + args.model);
    std::vector<char> response(kResponseBufferSize, 0);
    std::string messages = build_messages(args);
    std::string options = complete_options(args.max_tokens);
    int rc = cactus_complete(
        model,
        messages.c_str(),
        response.data(),
        response.size(),
        options.c_str(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0);
    cactus_destroy(model);
    std::cout << response.data() << "\n";
    return rc > 0 ? 0 : 2;
}

int run_transcribe(const Args& args, bool pcm) {
    if (args.audio.empty()) throw std::runtime_error("--audio is required");
    cactus_model_t model = cactus_init(args.model.c_str(), nullptr, false);
    if (!model) throw std::runtime_error("cactus_init failed for " + args.model);
    std::vector<char> response(kResponseBufferSize, 0);
    std::vector<uint8_t> pcm_data;
    const char* audio_path = args.audio.c_str();
    const uint8_t* pcm_buffer = nullptr;
    size_t pcm_size = 0;
    if (pcm) {
        pcm_data = read_wav_pcm16(args.audio);
        audio_path = nullptr;
        pcm_buffer = pcm_data.data();
        pcm_size = pcm_data.size();
    }
    int rc = cactus_transcribe(
        model,
        audio_path,
        nullptr,
        response.data(),
        response.size(),
        "{\"max_tokens\":200,\"telemetry_enabled\":false,\"auto_handoff\":false}",
        nullptr,
        nullptr,
        pcm_buffer,
        pcm_size);
    cactus_destroy(model);
    std::cout << response.data() << "\n";
    return rc > 0 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.command == "e002-tokens") return run_e002_tokens(args);
        if (args.command == "e002-preprocess") return run_e002_preprocess(args);
        if (args.command == "e002-vision") return run_e002_vision(args);
        if (args.command == "e002-vision-node-summary") return run_e002_vision_node_summary(args);
        if (args.command == "e002-vision-selected-nodes") return run_e002_vision_selected_nodes(args);
        if (args.command == "e002-audio-summary") return run_e002_audio_summary(args);
        if (args.command == "e002-audio-node-summary") return run_e002_audio_node_summary(args);
        if (args.command == "gemma-first-token") return run_gemma_first_token(args);
        if (args.command == "gemma-first-token-media-step") return run_gemma_first_token(args);
        if (args.command == "e003-preprocess-variants") return run_e003_preprocess_variants(args);
        if (args.command == "complete") return run_complete(args);
        if (args.command == "transcribe") return run_transcribe(args, false);
        if (args.command == "transcribe-pcm") return run_transcribe(args, true);
        throw std::runtime_error("unknown command: " + args.command);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
}
