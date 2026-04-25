#include "graph_param_io.h"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

void write_u32(std::ostream& out, uint32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_u64(std::ostream& out, uint64_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_i32(std::ostream& out, int32_t v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_f32(std::ostream& out, float v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

uint32_t read_u32(std::istream& in) {
    uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Unexpected EOF while reading uint32");
    return v;
}

uint64_t read_u64(std::istream& in) {
    uint64_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Unexpected EOF while reading uint64");
    return v;
}

int32_t read_i32(std::istream& in) {
    int32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Unexpected EOF while reading int32");
    return v;
}

float read_f32(std::istream& in) {
    float v = 0.0f;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error("Unexpected EOF while reading float");
    return v;
}

void write_size_vector(std::ostream& out, const std::vector<size_t>& values) {
    write_u32(out, static_cast<uint32_t>(values.size()));
    for (size_t v : values) write_u64(out, static_cast<uint64_t>(v));
}

std::vector<size_t> read_size_vector(std::istream& in) {
    uint32_t count = read_u32(in);
    std::vector<size_t> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) values.push_back(static_cast<size_t>(read_u64(in)));
    return values;
}

void write_u32_vector(std::ostream& out, const std::vector<uint32_t>& values) {
    write_u32(out, static_cast<uint32_t>(values.size()));
    for (uint32_t v : values) write_u32(out, v);
}

std::vector<uint32_t> read_u32_vector(std::istream& in) {
    uint32_t count = read_u32(in);
    std::vector<uint32_t> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) values.push_back(read_u32(in));
    return values;
}

void write_f32_vector(std::ostream& out, const std::vector<float>& values) {
    write_u32(out, static_cast<uint32_t>(values.size()));
    for (float v : values) write_f32(out, v);
}

std::vector<float> read_f32_vector(std::istream& in) {
    uint32_t count = read_u32(in);
    std::vector<float> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) values.push_back(read_f32(in));
    return values;
}

enum class ParamField : uint32_t {
    Scalar = 1,
    Axis,
    NewShape,
    PretransposedRhs,
    Backend,
    SliceStart,
    SliceLength,
    Epsilon,
    NumGroups,
    IndexValue,
    Permutation,
    Scale,
    Theta,
    PositionOffset,
    WindowSize,
    IsCausal,
    AttentionMaskIsAdditive,
    LogitCap,
    TopK,
    Temperature,
    TopP,
    MinP,
    RepetitionPenalty,
    RandomSeed,
    BiasIndices,
    BiasValues,
    NormalizeRouting,
    NumExperts,
    NumExpertsPerTok,
    MoeGated,
    Activation,
    NumKvHeads,
    ChunkSize,
    DstHeight,
    DstWidth,
    AlignCorners,
    NumClasses,
    NumAltupInputs,
    KernelSize,
    Stride,
    Dilation,
    NumFftBins,
    CacheSeqLen,
    HeadDim,
    VHeadDim,
    CachedKeysInt8Ptr,
    CachedValuesInt8Ptr,
    CachedKScalesPtr,
    CachedVScalesPtr,
};

enum class FieldPersistence {
    Persistent,
    RuntimeOnly,
    Derived,
};

struct FieldSpec {
    ParamField field;
    FieldPersistence persistence;
};

using Schema = std::vector<FieldSpec>;

const Schema& op_schema(OpType op_type) {
    static const Schema kEmpty{};
    static const std::unordered_map<OpType, Schema> kSchemas = {
        {OpType::POW, {{ParamField::Scalar, FieldPersistence::Persistent}}},
        {OpType::SCALAR_ADD, {{ParamField::Scalar, FieldPersistence::Persistent}}},
        {OpType::SCALAR_SUBTRACT, {{ParamField::Scalar, FieldPersistence::Persistent}}},
        {OpType::SCALAR_MULTIPLY, {{ParamField::Scalar, FieldPersistence::Persistent}}},
        {OpType::SCALAR_DIVIDE, {{ParamField::Scalar, FieldPersistence::Persistent}}},
        {OpType::SOFTMAX, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::SUM, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::MEAN, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::VARIANCE, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::MIN, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::MAX, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::INDEX, {{ParamField::Axis, FieldPersistence::Persistent}, {ParamField::IndexValue, FieldPersistence::Persistent}}},
        {OpType::CONCAT, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::CAT, {{ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::VIEW, {{ParamField::NewShape, FieldPersistence::Persistent}}},
        {OpType::RESHAPE, {{ParamField::NewShape, FieldPersistence::Persistent}}},
        {OpType::FLATTEN, {{ParamField::NewShape, FieldPersistence::Persistent}}},
        {OpType::MATMUL, {{ParamField::PretransposedRhs, FieldPersistence::Persistent}, {ParamField::Backend, FieldPersistence::Persistent}}},
        {OpType::TRANSPOSE, {{ParamField::Permutation, FieldPersistence::Persistent}, {ParamField::Backend, FieldPersistence::Persistent}}},
        {OpType::SLICE, {{ParamField::Axis, FieldPersistence::Persistent}, {ParamField::SliceStart, FieldPersistence::Persistent}, {ParamField::SliceLength, FieldPersistence::Persistent}}},
        {OpType::RMS_NORM, {{ParamField::Epsilon, FieldPersistence::Persistent}}},
        {OpType::LAYERNORM, {{ParamField::Epsilon, FieldPersistence::Persistent}}},
        {OpType::GROUPNORM, {{ParamField::Epsilon, FieldPersistence::Persistent}, {ParamField::NumGroups, FieldPersistence::Persistent}}},
        {OpType::BATCHNORM, {{ParamField::Epsilon, FieldPersistence::Persistent}, {ParamField::Axis, FieldPersistence::Persistent}}},
        {OpType::ROPE, {{ParamField::Theta, FieldPersistence::Persistent}, {ParamField::PositionOffset, FieldPersistence::Persistent}, {ParamField::Backend, FieldPersistence::Persistent}}},
        {OpType::ROPE_GPTJ, {{ParamField::Theta, FieldPersistence::Persistent}, {ParamField::PositionOffset, FieldPersistence::Persistent}, {ParamField::Scalar, FieldPersistence::Persistent}, {ParamField::Backend, FieldPersistence::Persistent}}},
        {OpType::TOPK, {{ParamField::TopK, FieldPersistence::Persistent}}},
        {OpType::ATTENTION, {{ParamField::Scale, FieldPersistence::Persistent}, {ParamField::PositionOffset, FieldPersistence::Persistent}, {ParamField::WindowSize, FieldPersistence::Persistent}, {ParamField::IsCausal, FieldPersistence::Persistent}, {ParamField::AttentionMaskIsAdditive, FieldPersistence::Persistent}, {ParamField::LogitCap, FieldPersistence::Persistent}, {ParamField::Backend, FieldPersistence::Persistent}}},
        {OpType::REL_POS_BIAS, {{ParamField::Scale, FieldPersistence::Persistent}}},
        {OpType::ATTENTION_INT8_HYBRID, {
            {ParamField::Scale, FieldPersistence::Persistent},
            {ParamField::PositionOffset, FieldPersistence::Persistent},
            {ParamField::WindowSize, FieldPersistence::Persistent},
            {ParamField::NumKvHeads, FieldPersistence::Persistent},
            {ParamField::CacheSeqLen, FieldPersistence::Persistent},
            {ParamField::HeadDim, FieldPersistence::Persistent},
            {ParamField::VHeadDim, FieldPersistence::Persistent},
            {ParamField::CachedKeysInt8Ptr, FieldPersistence::RuntimeOnly},
            {ParamField::CachedValuesInt8Ptr, FieldPersistence::RuntimeOnly},
            {ParamField::CachedKScalesPtr, FieldPersistence::RuntimeOnly},
            {ParamField::CachedVScalesPtr, FieldPersistence::RuntimeOnly},
        }},
        {OpType::MOE_LAYER, {{ParamField::Scalar, FieldPersistence::Persistent}, {ParamField::Epsilon, FieldPersistence::Persistent}, {ParamField::NormalizeRouting, FieldPersistence::Persistent}, {ParamField::NumExperts, FieldPersistence::Persistent}, {ParamField::NumExpertsPerTok, FieldPersistence::Persistent}, {ParamField::MoeGated, FieldPersistence::Persistent}, {ParamField::Activation, FieldPersistence::Persistent}}},
        {OpType::GATED_DELTANET_DECODE, {{ParamField::Scale, FieldPersistence::Persistent}, {ParamField::NumKvHeads, FieldPersistence::Persistent}}},
        {OpType::GATED_DELTANET_PREFILL, {{ParamField::Scale, FieldPersistence::Persistent}, {ParamField::NumKvHeads, FieldPersistence::Persistent}, {ParamField::ChunkSize, FieldPersistence::Persistent}}},
        {OpType::STFT, {{ParamField::Stride, FieldPersistence::Persistent}, {ParamField::NumFftBins, FieldPersistence::Persistent}}},
        {OpType::BILINEAR_INTERPOLATION, {{ParamField::DstHeight, FieldPersistence::Persistent}, {ParamField::DstWidth, FieldPersistence::Persistent}, {ParamField::AlignCorners, FieldPersistence::Persistent}}},
        {OpType::SCATTER_TOPK, {{ParamField::NumClasses, FieldPersistence::Persistent}}},
        {OpType::ALTUP_PREDICT, {{ParamField::NumAltupInputs, FieldPersistence::Persistent}}},
        {OpType::ALTUP_CORRECT, {{ParamField::NumAltupInputs, FieldPersistence::Persistent}}},
        {OpType::MAXPOOL1D, {{ParamField::KernelSize, FieldPersistence::Persistent}, {ParamField::Stride, FieldPersistence::Persistent}}},
        {OpType::CONV1D_CAUSAL, {{ParamField::Dilation, FieldPersistence::Persistent}}},
        {OpType::CONV1D_K3, {{ParamField::Stride, FieldPersistence::Persistent}}},
        {OpType::CONV1D_K7S3, {{ParamField::Stride, FieldPersistence::Persistent}}},
        {OpType::CONV1D, {{ParamField::Stride, FieldPersistence::Persistent}}},
        {OpType::SAMPLE, {{ParamField::Temperature, FieldPersistence::Persistent}, {ParamField::TopP, FieldPersistence::Persistent}, {ParamField::MinP, FieldPersistence::Persistent}, {ParamField::RepetitionPenalty, FieldPersistence::Persistent}, {ParamField::TopK, FieldPersistence::Persistent}, {ParamField::RandomSeed, FieldPersistence::Persistent}, {ParamField::BiasIndices, FieldPersistence::Persistent}, {ParamField::BiasValues, FieldPersistence::Persistent}}},
    };

    auto it = kSchemas.find(op_type);
    return it == kSchemas.end() ? kEmpty : it->second;
}

void write_field(std::ostream& out, ParamField field, const OpParams& params) {
    switch (field) {
        case ParamField::Scalar: write_f32(out, params.scalar); break;
        case ParamField::Axis: write_i32(out, static_cast<int32_t>(params.axis)); break;
        case ParamField::NewShape: write_size_vector(out, params.new_shape); break;
        case ParamField::PretransposedRhs: write_u32(out, params.pretransposed_rhs ? 1u : 0u); break;
        case ParamField::Backend: write_u32(out, static_cast<uint32_t>(params.backend)); break;
        case ParamField::SliceStart: write_u64(out, static_cast<uint64_t>(params.slice_start)); break;
        case ParamField::SliceLength: write_u64(out, static_cast<uint64_t>(params.slice_length)); break;
        case ParamField::Epsilon: write_f32(out, params.epsilon); break;
        case ParamField::NumGroups: write_u64(out, static_cast<uint64_t>(params.num_groups)); break;
        case ParamField::IndexValue: write_u64(out, static_cast<uint64_t>(params.index_value)); break;
        case ParamField::Permutation: write_size_vector(out, params.permutation); break;
        case ParamField::Scale: write_f32(out, params.scale); break;
        case ParamField::Theta: write_f32(out, params.theta); break;
        case ParamField::PositionOffset: write_u64(out, static_cast<uint64_t>(params.position_offset)); break;
        case ParamField::WindowSize: write_u64(out, static_cast<uint64_t>(params.window_size)); break;
        case ParamField::IsCausal: write_u32(out, params.is_causal ? 1u : 0u); break;
        case ParamField::AttentionMaskIsAdditive: write_u32(out, params.attention_mask_is_additive ? 1u : 0u); break;
        case ParamField::LogitCap: write_f32(out, params.logit_cap); break;
        case ParamField::TopK: write_u64(out, static_cast<uint64_t>(params.top_k)); break;
        case ParamField::Temperature: write_f32(out, params.temperature); break;
        case ParamField::TopP: write_f32(out, params.top_p); break;
        case ParamField::MinP: write_f32(out, params.min_p); break;
        case ParamField::RepetitionPenalty: write_f32(out, params.repetition_penalty); break;
        case ParamField::RandomSeed: write_u64(out, static_cast<uint64_t>(params.random_seed)); break;
        case ParamField::BiasIndices: write_u32_vector(out, params.bias_indices); break;
        case ParamField::BiasValues: write_f32_vector(out, params.bias_values); break;
        case ParamField::NormalizeRouting: write_u32(out, params.normalize_routing ? 1u : 0u); break;
        case ParamField::NumExperts: write_u64(out, static_cast<uint64_t>(params.num_experts)); break;
        case ParamField::NumExpertsPerTok: write_u64(out, static_cast<uint64_t>(params.num_experts_per_tok)); break;
        case ParamField::MoeGated: write_u32(out, params.moe_gated ? 1u : 0u); break;
        case ParamField::Activation: write_u32(out, static_cast<uint32_t>(params.activation)); break;
        case ParamField::NumKvHeads: write_u64(out, static_cast<uint64_t>(params.num_kv_heads)); break;
        case ParamField::ChunkSize: write_u64(out, static_cast<uint64_t>(params.chunk_size)); break;
        case ParamField::DstHeight: write_u64(out, static_cast<uint64_t>(params.dst_height)); break;
        case ParamField::DstWidth: write_u64(out, static_cast<uint64_t>(params.dst_width)); break;
        case ParamField::AlignCorners: write_u32(out, params.align_corners ? 1u : 0u); break;
        case ParamField::NumClasses: write_u64(out, static_cast<uint64_t>(params.num_classes)); break;
        case ParamField::NumAltupInputs: write_u64(out, static_cast<uint64_t>(params.num_altup_inputs)); break;
        case ParamField::KernelSize: write_u64(out, static_cast<uint64_t>(params.kernel_size)); break;
        case ParamField::Stride: write_u64(out, static_cast<uint64_t>(params.stride)); break;
        case ParamField::Dilation: write_u64(out, static_cast<uint64_t>(params.dilation)); break;
        case ParamField::NumFftBins: write_u64(out, static_cast<uint64_t>(params.num_fft_bins)); break;
        case ParamField::CacheSeqLen: write_u64(out, static_cast<uint64_t>(params.cache_seq_len)); break;
        case ParamField::HeadDim: write_u64(out, static_cast<uint64_t>(params.head_dim)); break;
        case ParamField::VHeadDim: write_u64(out, static_cast<uint64_t>(params.v_head_dim)); break;
        case ParamField::CachedKeysInt8Ptr:
        case ParamField::CachedValuesInt8Ptr:
        case ParamField::CachedKScalesPtr:
        case ParamField::CachedVScalesPtr:
            throw std::runtime_error("Attempted to serialize runtime-only pointer field");
    }
}

void read_field(std::istream& in, ParamField field, OpParams& params) {
    switch (field) {
        case ParamField::Scalar: params.scalar = read_f32(in); break;
        case ParamField::Axis: params.axis = static_cast<int>(read_i32(in)); break;
        case ParamField::NewShape: params.new_shape = read_size_vector(in); break;
        case ParamField::PretransposedRhs: params.pretransposed_rhs = (read_u32(in) != 0); break;
        case ParamField::Backend: {
            uint32_t backend_val = read_u32(in);
            if (backend_val > static_cast<uint32_t>(ComputeBackend::NPU)) {
                throw std::runtime_error("Graph file corrupted: invalid backend");
            }
            params.backend = static_cast<ComputeBackend>(backend_val);
            break;
        }
        case ParamField::SliceStart: params.slice_start = static_cast<size_t>(read_u64(in)); break;
        case ParamField::SliceLength: params.slice_length = static_cast<size_t>(read_u64(in)); break;
        case ParamField::Epsilon: params.epsilon = read_f32(in); break;
        case ParamField::NumGroups: params.num_groups = static_cast<size_t>(read_u64(in)); break;
        case ParamField::IndexValue: params.index_value = static_cast<size_t>(read_u64(in)); break;
        case ParamField::Permutation: params.permutation = read_size_vector(in); break;
        case ParamField::Scale: params.scale = read_f32(in); break;
        case ParamField::Theta: params.theta = read_f32(in); break;
        case ParamField::PositionOffset: params.position_offset = static_cast<size_t>(read_u64(in)); break;
        case ParamField::WindowSize: params.window_size = static_cast<size_t>(read_u64(in)); break;
        case ParamField::IsCausal: params.is_causal = (read_u32(in) != 0); break;
        case ParamField::AttentionMaskIsAdditive: params.attention_mask_is_additive = (read_u32(in) != 0); break;
        case ParamField::LogitCap: params.logit_cap = read_f32(in); break;
        case ParamField::TopK: params.top_k = static_cast<size_t>(read_u64(in)); break;
        case ParamField::Temperature: params.temperature = read_f32(in); break;
        case ParamField::TopP: params.top_p = read_f32(in); break;
        case ParamField::MinP: params.min_p = read_f32(in); break;
        case ParamField::RepetitionPenalty: params.repetition_penalty = read_f32(in); break;
        case ParamField::RandomSeed: params.random_seed = static_cast<size_t>(read_u64(in)); break;
        case ParamField::BiasIndices: params.bias_indices = read_u32_vector(in); break;
        case ParamField::BiasValues: params.bias_values = read_f32_vector(in); break;
        case ParamField::NormalizeRouting: params.normalize_routing = (read_u32(in) != 0); break;
        case ParamField::NumExperts: params.num_experts = static_cast<size_t>(read_u64(in)); break;
        case ParamField::NumExpertsPerTok: params.num_experts_per_tok = static_cast<size_t>(read_u64(in)); break;
        case ParamField::MoeGated: params.moe_gated = (read_u32(in) != 0); break;
        case ParamField::Activation: params.activation = static_cast<Activation>(read_u32(in)); break;
        case ParamField::NumKvHeads: params.num_kv_heads = static_cast<size_t>(read_u64(in)); break;
        case ParamField::ChunkSize: params.chunk_size = static_cast<size_t>(read_u64(in)); break;
        case ParamField::DstHeight: params.dst_height = static_cast<size_t>(read_u64(in)); break;
        case ParamField::DstWidth: params.dst_width = static_cast<size_t>(read_u64(in)); break;
        case ParamField::AlignCorners: params.align_corners = (read_u32(in) != 0); break;
        case ParamField::NumClasses: params.num_classes = static_cast<size_t>(read_u64(in)); break;
        case ParamField::NumAltupInputs: params.num_altup_inputs = static_cast<size_t>(read_u64(in)); break;
        case ParamField::KernelSize: params.kernel_size = static_cast<size_t>(read_u64(in)); break;
        case ParamField::Stride: params.stride = static_cast<size_t>(read_u64(in)); break;
        case ParamField::Dilation: params.dilation = static_cast<size_t>(read_u64(in)); break;
        case ParamField::NumFftBins: params.num_fft_bins = static_cast<size_t>(read_u64(in)); break;
        case ParamField::CacheSeqLen: params.cache_seq_len = static_cast<size_t>(read_u64(in)); break;
        case ParamField::HeadDim: params.head_dim = static_cast<size_t>(read_u64(in)); break;
        case ParamField::VHeadDim: params.v_head_dim = static_cast<size_t>(read_u64(in)); break;
        case ParamField::CachedKeysInt8Ptr:
        case ParamField::CachedValuesInt8Ptr:
        case ParamField::CachedKScalesPtr:
        case ParamField::CachedVScalesPtr:
            throw std::runtime_error("Graph file corrupted: runtime-only field serialized");
    }
}

} // namespace

namespace GraphParamIO {

void write_op_params(std::ostream& out, OpType op_type, const OpParams& params) {
    const auto& schema = op_schema(op_type);
    std::vector<ParamField> fields;
    fields.reserve(schema.size());
    for (const auto& spec : schema) {
        if (spec.persistence == FieldPersistence::Persistent) {
            fields.push_back(spec.field);
        }
    }

    write_u32(out, static_cast<uint32_t>(fields.size()));
    for (ParamField field : fields) {
        write_u32(out, static_cast<uint32_t>(field));
        write_field(out, field, params);
    }
}

void read_op_params(std::istream& in, OpType op_type, OpParams& params) {
    uint32_t field_count = read_u32(in);
    const auto& schema = op_schema(op_type);
    std::unordered_set<uint32_t> allowed;
    for (const auto& spec : schema) {
        if (spec.persistence == FieldPersistence::Persistent) {
            allowed.insert(static_cast<uint32_t>(spec.field));
        }
    }

    for (uint32_t i = 0; i < field_count; ++i) {
        uint32_t raw_field = read_u32(in);
        if (allowed.count(raw_field) == 0) {
            throw std::runtime_error("Graph file corrupted: field not allowed for op");
        }
        read_field(in, static_cast<ParamField>(raw_field), params);
    }
}

} // namespace GraphParamIO
