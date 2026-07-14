#include "../cactus_graph.h"
#include "param_io.h"
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <filesystem>

namespace {
    constexpr uint32_t fourcc(char a, char b, char c, char d) {
        return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
               (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
               (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
    }

    constexpr uint32_t CACTUS_MAGIC = 0x54434143;
    constexpr uint32_t CACTUS_GRAPH_MAGIC = fourcc('C', 'G', 'R', 'F');
    constexpr uint32_t CACTUS_GRAPH_VERSION_LEGACY = 4;
    constexpr uint32_t CACTUS_GRAPH_VERSION_EMBEDDED_INPUTS = 5;
    constexpr uint32_t CACTUS_GRAPH_VERSION_DYNAMIC_SHAPES = 6;
    constexpr uint32_t FLAG_ORTHOGONAL_ROTATION = 1 << 1;
    constexpr uint32_t FLAG_INTERLEAVED_4ROW = 1 << 2;
    constexpr uint32_t FLAG_EXTENDED_SHAPE = 1 << 4;
    constexpr size_t HEADER_SIZE = 84;

    inline size_t align_offset(size_t offset, size_t alignment) {
        size_t remainder = offset % alignment;
        if (remainder == 0) return offset;
        return offset + (alignment - remainder);
    }

    std::string resolve_quantized_weight_file(const std::string& filename) {
        constexpr const char* suffix = ".weights";
        if (filename.size() <= std::strlen(suffix) ||
            filename.compare(filename.size() - std::strlen(suffix), std::strlen(suffix), suffix) != 0) {
            return filename;
        }

        const std::string stem = filename.substr(0, filename.size() - std::strlen(suffix));
        const std::string cq4 = stem + ".cq4.weights";
        if (std::filesystem::exists(cq4)) {
            return cq4;
        }
        const std::string cq2 = stem + ".cq2.weights";
        if (std::filesystem::exists(cq2)) {
            return cq2;
        }
        return filename;
    }

    inline void write_u32(std::ostream& out, uint32_t v) {
      out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    inline void write_u64(std::ostream& out, uint64_t v) {
      out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    void write_size_vector(std::ostream& out, const std::vector<size_t>& values) {
      uint32_t size = static_cast<uint32_t>(values.size());
      write_u32(out, size);
      for (size_t v : values) {
        write_u64(out, static_cast<uint64_t>(v));
      }
    }

    void write_u32_vector(std::ostream& out, const std::vector<uint32_t>& values) {
      uint32_t size = static_cast<uint32_t>(values.size());
      write_u32(out, size);
      for (uint32_t v : values) {
        write_u32(out, v);
      }
    }

    std::vector<uint32_t> compute_leaf_outputs(const GraphFile::SerializedGraph& sg) {
      std::unordered_set<uint32_t> node_ids;
      std::unordered_set<uint32_t> referenced;
      node_ids.reserve(sg.nodes.size());
      referenced.reserve(sg.nodes.size());

      for (const auto& node : sg.nodes) {
        node_ids.insert(node.index);
      }

      for (const auto& node : sg.nodes) {
        for (uint32_t input_idx : node.inputs) {
          if (node_ids.find(input_idx) == node_ids.end()) {
            throw std::runtime_error("Graph save failed: input node id not found");
          }
          referenced.insert(input_idx);
        }
      }

      std::vector<uint32_t> outputs;
      outputs.reserve(sg.nodes.size());

      for (const auto& node : sg.nodes) {
        if (referenced.find(node.index) == referenced.end()) {
          outputs.push_back(node.index);
        }
      }

      return outputs;
    }

    void write_serialized_graph(std::ostream& out, const GraphFile::SerializedGraph& sg) {
      write_u32(out, sg.header.magic);
      write_u32(out, sg.header.version);
      write_u32(out, sg.header.node_count);
      write_u32(out, sg.header.flags);

      write_u32_vector(out, sg.graph_inputs);
      write_u32_vector(out, sg.graph_outputs);

      for (const auto& node : sg.nodes) {
        write_u32(out, node.index);
        write_u32(out, static_cast<uint32_t>(node.op_type));
        write_u32_vector(out, node.inputs);
        write_size_vector(out, node.output_shape);
        write_u32(out, static_cast<uint32_t>(node.precision));
        GraphParamIO::write_op_params(out, node.op_type, node.params);
        if (sg.header.version >= CACTUS_GRAPH_VERSION_EMBEDDED_INPUTS) {
            write_u32(out, node.has_embedded_data ? 1u : 0u);
            write_u64(out, static_cast<uint64_t>(node.embedded_data.size()));
            if (!node.embedded_data.empty()) {
                out.write(
                    reinterpret_cast<const char*>(node.embedded_data.data()),
                    static_cast<std::streamsize>(node.embedded_data.size())
                );
            }
        }
        if (sg.header.version >= CACTUS_GRAPH_VERSION_DYNAMIC_SHAPES) {
            write_u32(out, static_cast<uint32_t>(node.dynamic_mask.size()));
            if (!node.dynamic_mask.empty()) {
                out.write(
                    reinterpret_cast<const char*>(node.dynamic_mask.data()),
                    static_cast<std::streamsize>(node.dynamic_mask.size())
                );
            }
        }
      }

      if (!out) {
        throw std::runtime_error("Error writing serialized graph");
      }
    }

    // read helpers
    uint32_t read_u32(std::istream& in) {
        uint32_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading uint32");
        }
        return v;
    }

    uint64_t read_u64(std::istream& in) {
        uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        if (!in) {
            throw std::runtime_error("Unexpected EOF while reading uint64");
        }
        return v;
    }

    std::vector<uint32_t> read_u32_vector(std::istream& in) {
        uint32_t count = read_u32(in);
        std::vector<uint32_t> values;
        values.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            values.push_back(read_u32(in));
        }
        return values;
    }

    std::vector<size_t> read_size_vector(std::istream& in) {
        uint32_t count = read_u32(in);
        std::vector<size_t> values;
        values.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            values.push_back(static_cast<size_t>(read_u64(in)));
        }
        return values;
    }

    bool is_binary_broadcast_op(OpType op_type) {
        switch (op_type) {
            case OpType::ADD:
            case OpType::ADD_CLIPPED:
            case OpType::SUBTRACT:
            case OpType::MULTIPLY:
            case OpType::DIVIDE:
            case OpType::NOT_EQUAL:
                return true;
            default:
                return false;
        }
    }

    void populate_derived_params(CactusGraph& graph,
                                 const GraphFile::NodeEntry& node_entry,
                                 const std::vector<size_t>& runtime_inputs,
                                 OpParams& params) {
        if (is_binary_broadcast_op(node_entry.op_type)) {
            if (runtime_inputs.size() != 2) {
                throw std::runtime_error("Graph file corrupted: binary op missing inputs");
            }
            const auto& lhs = graph.get_output_buffer(runtime_inputs[0]);
            const auto& rhs = graph.get_output_buffer(runtime_inputs[1]);
            params.broadcast_info = BroadcastInfo::compute(lhs.shape, rhs.shape);
        }
    }

    GraphFile::GraphHeader read_graph_header(std::istream& in) {
        GraphFile::GraphHeader header;
        header.magic = read_u32(in);
        header.version = read_u32(in);
        header.node_count = read_u32(in);
        header.flags = read_u32(in);

        if (header.magic != CACTUS_GRAPH_MAGIC) {
            throw std::runtime_error("Invalid graph file: bad magic");
        }
        if (header.version != CACTUS_GRAPH_VERSION_LEGACY &&
            header.version != CACTUS_GRAPH_VERSION_EMBEDDED_INPUTS &&
            header.version != CACTUS_GRAPH_VERSION_DYNAMIC_SHAPES) {
            throw std::runtime_error("Unsupported graph file version: " +
                std::to_string(header.version));
        }

        return header;
    }

    GraphFile::NodeEntry read_node_entry(std::istream& in, uint32_t graph_version) {
        GraphFile::NodeEntry node;
        node.index = read_u32(in);
        uint32_t op_type_val = read_u32(in);
        if (op_type_val > static_cast<uint32_t>(OpType::CONV_CACHE_INITIALIZE)) {
            throw std::runtime_error("Graph file corrupted: invalid op type");
        }
        node.op_type = static_cast<OpType>(op_type_val);
        node.inputs = read_u32_vector(in);
        node.output_shape = read_size_vector(in);
        uint32_t precision_val = read_u32(in);
        if (precision_val > static_cast<uint32_t>(Precision::CQ4)) {
            throw std::runtime_error("Graph file corrupted: invalid precision");
        }
        node.precision = static_cast<Precision>(precision_val);
        GraphParamIO::read_op_params(in, node.op_type, node.params);
        if (graph_version >= CACTUS_GRAPH_VERSION_EMBEDDED_INPUTS) {
            node.has_embedded_data = read_u32(in) != 0;
            uint64_t data_size = read_u64(in);
            if (data_size > 0) {
                node.embedded_data.resize(static_cast<size_t>(data_size));
                in.read(
                    reinterpret_cast<char*>(node.embedded_data.data()),
                    static_cast<std::streamsize>(node.embedded_data.size())
                );
                if (!in) {
                    throw std::runtime_error("Unexpected EOF while reading embedded graph input data");
                }
            }
        }
        if (graph_version >= CACTUS_GRAPH_VERSION_DYNAMIC_SHAPES) {
            uint32_t mask_size = read_u32(in);
            if (mask_size > 0) {
                node.dynamic_mask.resize(mask_size);
                in.read(reinterpret_cast<char*>(node.dynamic_mask.data()),
                        static_cast<std::streamsize>(mask_size));
                if (!in) {
                    throw std::runtime_error("Unexpected EOF while reading dynamic mask");
                }
            }
        }
        return node;
    }
    

} // namespace

void CactusGraph::save(const std::string& path) {
    GraphFile::save_graph(*this, path);
}

void CactusGraph::mark_embedded_input(size_t node_id) {
    auto it = node_index_map_.find(node_id);
    if (it == node_index_map_.end()) {
        throw std::out_of_range("Unknown input node id: " + std::to_string(node_id));
    }
    const auto& node = *nodes_[it->second];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only embed input nodes");
    }
    embedded_input_node_ids_.insert(node_id);
}

bool CactusGraph::is_embedded_input(size_t node_id) const {
    return embedded_input_node_ids_.find(node_id) != embedded_input_node_ids_.end();
}

CactusGraph CactusGraph::from_serialized(const GraphFile::SerializedGraph& sg) {
    CactusGraph graph;
    std::vector<size_t> runtime_ids;
    runtime_ids.reserve(sg.nodes.size());
    std::unordered_map<uint32_t, size_t> serialized_id_to_runtime_id;
    serialized_id_to_runtime_id.reserve(sg.nodes.size());

    if (sg.nodes.size() != sg.header.node_count) {
        throw std::runtime_error("Graph file corrupted: node count mismatch");
    }

    bool dense_legacy_indices = true;
    for (size_t i = 0; i < sg.nodes.size(); ++i) {
        if (sg.nodes[i].index != i) {
            dense_legacy_indices = false;
            break;
        }
    }

    for (size_t i = 0; i < sg.nodes.size(); ++i) {
        const auto& node_entry = sg.nodes[i];

        std::vector<size_t> runtime_inputs;
        runtime_inputs.reserve(node_entry.inputs.size());

        for (uint32_t serialized_input_id : node_entry.inputs) {
            if (dense_legacy_indices) {
                if (serialized_input_id >= runtime_ids.size()) {
                    throw std::runtime_error(
                        "Graph file corrupted: input refers to a node that has not been reconstructed yet"
                    );
                }
                runtime_inputs.push_back(runtime_ids[serialized_input_id]);
            } else {
                auto it = serialized_id_to_runtime_id.find(serialized_input_id);
                if (it == serialized_id_to_runtime_id.end()) {
                    throw std::runtime_error(
                        "Graph file corrupted: input refers to a node that has not been reconstructed yet"
                    );
                }
                runtime_inputs.push_back(it->second);
            }
        }

        size_t new_node_id = 0;

        if (!dense_legacy_indices) {
            graph.next_node_id_ = static_cast<size_t>(node_entry.index);
        }

        if (node_entry.op_type == OpType::INPUT) {
            new_node_id = graph.input(node_entry.output_shape, node_entry.precision);
            if (node_entry.has_embedded_data) {
                const auto& buffer = graph.get_output_buffer(new_node_id);
                if (node_entry.embedded_data.size() != buffer.byte_size) {
                    throw std::runtime_error(
                        "Graph file corrupted: embedded input byte-size mismatch for node " +
                        std::to_string(node_entry.index)
                    );
                }
                graph.set_input(new_node_id, node_entry.embedded_data.data(), node_entry.precision);
                graph.mark_embedded_input(new_node_id);
            }
        }
        else {
            OpParams params = node_entry.params;
            params.output_precision = node_entry.precision;
            populate_derived_params(graph, node_entry, runtime_inputs, params);
            new_node_id = graph.add_node(node_entry.op_type, runtime_inputs, node_entry.output_shape, params);

            if (node_entry.op_type == OpType::PERSISTENT
                || node_entry.op_type == OpType::KV_CACHE_STATE
                || node_entry.op_type == OpType::CONV_CACHE_STATE
                || node_entry.op_type == OpType::RECURRENT_CACHE_STATE) {
                graph.persistent_node_ids_.insert(new_node_id);
            }
        }
        if (!node_entry.dynamic_mask.empty()) {
            graph.nodes_[graph.node_index_map_.at(new_node_id)]->output_buffer.dynamic_dims = node_entry.dynamic_mask;
            graph.has_dynamic_shapes_ = true;
        }
        runtime_ids.push_back(new_node_id);
        serialized_id_to_runtime_id[node_entry.index] = new_node_id;
    }
    return graph;
}

CactusGraph CactusGraph::load(const std::string& path) {
    GraphFile::SerializedGraph sg = GraphFile::load_graph(path);
    return from_serialized(sg);
}

size_t CactusGraph::mmap_embeddings(const std::string& filename) {
    const std::string resolved_filename = resolve_quantized_weight_file(filename);
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(resolved_filename);

    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Memory-mapped embeddings must be 2D [vocab_size, embedding_dim]");
    }

    Precision precision = mapped_file->precision();

    size_t node_id = input(shape, precision);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);

    auto& buffer = nodes_[node_index_map_.at(node_id)]->output_buffer;
    if (PrecisionTraits::is_cq(precision) && mapped_file->group_size() > 0) {
        buffer.group_size = mapped_file->group_size();
        buffer.num_groups = mapped_file->num_groups();

        const char* scales_base = static_cast<const char*>(mapped_file->scales_data());
        uint32_t bits = PrecisionTraits::cq_bits(precision);
        uint32_t cb_size = 1u << bits;
        uint32_t gs = static_cast<uint32_t>(mapped_file->group_size());
        uint32_t K = gs * static_cast<uint32_t>(mapped_file->num_groups());
        uint32_t N = static_cast<uint32_t>(shape.size() >= 2 ? shape[0] : 1);

        size_t off = 0;
        buffer.cq_codebook = reinterpret_cast<const __fp16*>(scales_base + off);
        off += cb_size * sizeof(__fp16);
        buffer.cq_input_scale = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_input_scale_recip = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_norms = reinterpret_cast<const __fp16*>(scales_base + off);
        off += static_cast<size_t>(N) * mapped_file->num_groups() * sizeof(__fp16);

        if (mapped_file->is_orthogonal_rotation()) {
            buffer.cq_rotation = reinterpret_cast<const __fp16*>(scales_base + off);
            buffer.cq_flags = CACTUS_QUANT_FLAG_ORTHOGONAL;
        } else {
            buffer.cq_left_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_right_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_permutation = reinterpret_cast<const uint32_t*>(scales_base + off);
            buffer.cq_flags = 0;
        }
        if (mapped_file->is_interleaved_4row()) {
            buffer.cq_flags |= CACTUS_QUANT_FLAG_INTERLEAVED_4ROW;
        }
    } else if (precision == Precision::INT8 && mapped_file->group_size() > 0) {
        buffer.group_size = mapped_file->group_size();
        buffer.num_groups = mapped_file->num_groups();
        buffer.set_activation_scales(const_cast<void*>(mapped_file->scales_data()), shape.empty() ? 0 : shape[0]);
    }

    size_t file_idx = mapped_files_.size();
    mapped_files_.push_back(std::move(mapped_file));
    node_to_mapped_file_[node_id] = file_idx;
    weight_cache_[filename] = node_id;
    return node_id;
}

size_t CactusGraph::mmap_weights(const std::string& filename) {
    const std::string resolved_filename = resolve_quantized_weight_file(filename);
    auto it = weight_cache_.find(resolved_filename);
    if (it != weight_cache_.end()) {
        return it->second;
    }

    auto mapped_file = std::make_unique<GraphFile::MappedFile>(resolved_filename);

    const auto& shape = mapped_file->shape();
    Precision precision = mapped_file->precision();

    size_t node_id = input(shape, precision);
    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);

    if (PrecisionTraits::is_cq(precision) && mapped_file->group_size() > 0) {
        auto& buffer = nodes_[node_index_map_.at(node_id)]->output_buffer;
        buffer.group_size = mapped_file->group_size();
        buffer.num_groups = mapped_file->num_groups();


        const char* scales_base = static_cast<const char*>(mapped_file->scales_data());
        uint32_t bits = PrecisionTraits::cq_bits(precision);
        uint32_t cb_size = 1u << bits;
        uint32_t gs = static_cast<uint32_t>(mapped_file->group_size());
        uint32_t K = gs * static_cast<uint32_t>(mapped_file->num_groups());
        uint32_t N = static_cast<uint32_t>(shape.size() >= 2 ? shape[0] : 1);

        size_t off = 0;
        buffer.cq_codebook = reinterpret_cast<const __fp16*>(scales_base + off);
        off += cb_size * sizeof(__fp16);
        buffer.cq_input_scale = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_input_scale_recip = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_norms = reinterpret_cast<const __fp16*>(scales_base + off);
        off += static_cast<size_t>(N) * mapped_file->num_groups() * sizeof(__fp16);

        if (mapped_file->is_orthogonal_rotation()) {
            buffer.cq_rotation = reinterpret_cast<const __fp16*>(scales_base + off);
            buffer.cq_flags = CACTUS_QUANT_FLAG_ORTHOGONAL;
        } else {
            buffer.cq_left_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_right_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_permutation = reinterpret_cast<const uint32_t*>(scales_base + off);
            buffer.cq_flags = 0;
        }
        if (mapped_file->is_interleaved_4row()) {
            buffer.cq_flags |= CACTUS_QUANT_FLAG_INTERLEAVED_4ROW;
        }
    }

    size_t file_idx = mapped_files_.size();
    mapped_files_.push_back(std::move(mapped_file));
    node_to_mapped_file_[node_id] = file_idx;
    weight_cache_[resolved_filename] = node_id;
    return node_id;
}

void CactusGraph::bind_mmap_weights(size_t node_id, const std::string& filename) {
    const std::string resolved_filename = resolve_quantized_weight_file(filename);
    auto node_it = node_index_map_.find(node_id);
    if (node_it == node_index_map_.end()) {
        throw std::out_of_range("Unknown input node id: " + std::to_string(node_id));
    }

    auto& node = *nodes_[node_it->second];
    if (node.op_type != OpType::INPUT) {
        throw std::invalid_argument("Can only bind mmap weights to input nodes");
    }

    auto mapped_file = std::make_unique<GraphFile::MappedFile>(resolved_filename);
    const auto& shape = mapped_file->shape();
    Precision precision = mapped_file->precision();
    auto& buffer = node.output_buffer;
    if (buffer.shape != shape) {
        throw std::runtime_error("mmap weight shape mismatch for node " + std::to_string(node_id));
    }
    if (buffer.precision != precision) {
        throw std::runtime_error("mmap weight precision mismatch for node " + std::to_string(node_id));
    }

    set_external_input(node_id, const_cast<void*>(mapped_file->data()), precision);
    buffer.group_size = 0;
    buffer.num_groups = 0;
    buffer.activation_scales_data = nullptr;
    buffer.owned_activation_scales.reset();
    buffer.num_rows_for_activation_scales = 0;
    buffer.cq_codebook = nullptr;
    buffer.cq_input_scale = nullptr;
    buffer.cq_input_scale_recip = nullptr;
    buffer.cq_norms = nullptr;
    buffer.cq_left_signs = nullptr;
    buffer.cq_right_signs = nullptr;
    buffer.cq_permutation = nullptr;
    buffer.cq_rotation = nullptr;
    buffer.cq_flags = 0;

    if (PrecisionTraits::is_cq(precision) && mapped_file->group_size() > 0) {
        buffer.group_size = mapped_file->group_size();
        buffer.num_groups = mapped_file->num_groups();

        const char* scales_base = static_cast<const char*>(mapped_file->scales_data());
        uint32_t bits = PrecisionTraits::cq_bits(precision);
        uint32_t cb_size = 1u << bits;
        uint32_t gs = static_cast<uint32_t>(mapped_file->group_size());
        uint32_t K = gs * static_cast<uint32_t>(mapped_file->num_groups());
        uint32_t N = static_cast<uint32_t>(shape.size() >= 2 ? shape[0] : 1);

        size_t off = 0;
        buffer.cq_codebook = reinterpret_cast<const __fp16*>(scales_base + off);
        off += cb_size * sizeof(__fp16);
        buffer.cq_input_scale = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_input_scale_recip = reinterpret_cast<const __fp16*>(scales_base + off);
        off += K * sizeof(__fp16);
        buffer.cq_norms = reinterpret_cast<const __fp16*>(scales_base + off);
        off += static_cast<size_t>(N) * mapped_file->num_groups() * sizeof(__fp16);

        if (mapped_file->is_orthogonal_rotation()) {
            buffer.cq_rotation = reinterpret_cast<const __fp16*>(scales_base + off);
            buffer.cq_flags = CACTUS_QUANT_FLAG_ORTHOGONAL;
        } else {
            buffer.cq_left_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_right_signs = reinterpret_cast<const int8_t*>(scales_base + off);
            off += gs;
            buffer.cq_permutation = reinterpret_cast<const uint32_t*>(scales_base + off);
            buffer.cq_flags = 0;
        }
        if (mapped_file->is_interleaved_4row()) {
            buffer.cq_flags |= CACTUS_QUANT_FLAG_INTERLEAVED_4ROW;
        }
    } else if (precision == Precision::INT8 && mapped_file->group_size() > 0) {
        buffer.group_size = mapped_file->group_size();
        buffer.num_groups = mapped_file->num_groups();
        buffer.set_activation_scales(const_cast<void*>(mapped_file->scales_data()), shape.empty() ? 0 : shape[0]);
    }

    size_t file_idx = mapped_files_.size();
    mapped_files_.push_back(std::move(mapped_file));
    node_to_mapped_file_[node_id] = file_idx;
    weight_cache_[resolved_filename] = node_id;
}

void CactusGraph::release_weight_pages(size_t node_id) {
    auto it = node_to_mapped_file_.find(node_id);
    if (it != node_to_mapped_file_.end() && it->second < mapped_files_.size()) {
        mapped_files_[it->second]->release_pages();
    }
}

void CactusGraph::prefetch_weight_pages(size_t node_id) {
    auto it = node_to_mapped_file_.find(node_id);
    if (it != node_to_mapped_file_.end() && it->second < mapped_files_.size()) {
        mapped_files_[it->second]->prefetch_pages();
    }
}

void CactusGraph::release_all_weight_pages() {
    for (auto& mf : mapped_files_) {
        if (mf) mf->release_pages();
    }
}

size_t CactusGraph::embedding(const std::string& filename, size_t indices, ComputeBackend backend) {
    auto mapped_file = std::make_unique<GraphFile::MappedFile>(filename);

    const auto& shape = mapped_file->shape();
    if (shape.size() != 2) {
        throw std::runtime_error("Embedding file must contain 2D tensor [vocab_size, hidden_dim]");
    }

    Precision precision = mapped_file->precision();
    size_t embeddings_node = input(shape, precision);
    set_external_input(embeddings_node, const_cast<void*>(mapped_file->data()), precision);

    mapped_files_.push_back(std::move(mapped_file));

    const auto& idx_shape = get_output_buffer(indices).shape;
    std::vector<size_t> output_shape = idx_shape;
    output_shape.push_back(shape[1]);

    OpParams params;
    params.output_precision = Precision::FP16;
    return tag_backend(add_node(OpType::EMBEDDING, {embeddings_node, indices}, output_shape, params), backend);
}


namespace GraphFile {

void save_graph(const CactusGraph& graph,
                const std::string& filename) {

  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  SerializedGraph sg;
  sg.header.magic = CACTUS_GRAPH_MAGIC;
  bool any_dynamic = false;
  for (const auto& node : graph.nodes_) {
    if (!node->output_buffer.dynamic_dims.empty()) { any_dynamic = true; break; }
  }
  sg.header.version = any_dynamic ? CACTUS_GRAPH_VERSION_DYNAMIC_SHAPES
                                  : CACTUS_GRAPH_VERSION_EMBEDDED_INPUTS;
  sg.header.node_count = static_cast<uint32_t>(graph.nodes_.size());
  sg.header.flags = 0;

  sg.nodes.reserve(graph.nodes_.size());

  for (uint32_t i = 0; i < graph.nodes_.size(); ++i) {
    const auto& node = graph.nodes_[i];

    NodeEntry entry;
    entry.index = static_cast<uint32_t>(node->id);
    entry.op_type = node->op_type;
    entry.output_shape = node->output_buffer.shape;
    entry.precision = node->output_buffer.precision;
    entry.params = node->params;
    entry.dynamic_mask = node->output_buffer.dynamic_dims;

    entry.inputs.reserve(node->input_ids.size());
    for (size_t input_id : node->input_ids) {
      entry.inputs.push_back(static_cast<uint32_t>(input_id));
    }

    if (node->op_type == OpType::INPUT && graph.is_embedded_input(node->id)) {
      if (node->output_buffer.external_data != nullptr) {
        throw std::runtime_error(
            "Graph save failed: embedded input node " + std::to_string(node->id) +
            " is backed by external data"
        );
      }
      const void* data = node->output_buffer.get_data();
      if (data == nullptr && node->output_buffer.byte_size > 0) {
        throw std::runtime_error(
            "Graph save failed: embedded input node " + std::to_string(node->id) +
            " has no materialized data"
        );
      }
      entry.has_embedded_data = true;
      if (node->output_buffer.byte_size > 0) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        entry.embedded_data.assign(bytes, bytes + node->output_buffer.byte_size);
      }
    }

    if (node->op_type == OpType::INPUT && !entry.has_embedded_data) {
      sg.graph_inputs.push_back(entry.index);
    }

    sg.nodes.push_back(std::move(entry));
  }

  sg.graph_outputs = compute_leaf_outputs(sg);

  write_serialized_graph(out, sg);

  if (!out) {
    throw std::runtime_error("Error writing graph data to file: " + filename);
  }
}

SerializedGraph load_graph(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    SerializedGraph sg;
    sg.header = read_graph_header(in);
    sg.graph_inputs = read_u32_vector(in);
    sg.graph_outputs = read_u32_vector(in);
    sg.nodes.reserve(sg.header.node_count);
    for (uint32_t i = 0; i < sg.header.node_count; ++i) {
        sg.nodes.push_back(read_node_entry(in, sg.header.version));
    }

    if (sg.nodes.size() != sg.header.node_count) {
        throw std::runtime_error("Graph file corrupted: node count mismatch");
    }

    bool dense_legacy_indices = true;
    std::unordered_set<uint32_t> node_ids;
    node_ids.reserve(sg.nodes.size());
    for (uint32_t i = 0; i < sg.nodes.size(); ++i) {
        const auto& node = sg.nodes[i];
        if (node.index != i) {
            dense_legacy_indices = false;
        }
        node_ids.insert(node.index);
    }

    for (const auto& node : sg.nodes) {
        for (uint32_t input_idx : node.inputs) {
            if (dense_legacy_indices) {
                if (input_idx >= sg.nodes.size()) {
                    throw std::runtime_error("Graph file corrupted: input index out of range");
                }
            } else if (node_ids.find(input_idx) == node_ids.end()) {
                throw std::runtime_error("Graph file corrupted: input node id out of range");
            }
        }
    }

    for (uint32_t input_idx : sg.graph_inputs) {
        if (dense_legacy_indices) {
            if (input_idx >= sg.nodes.size()) {
                throw std::runtime_error("Graph file corrupted: graph input index out of range");
            }
        } else if (node_ids.find(input_idx) == node_ids.end()) {
            throw std::runtime_error("Graph file corrupted: graph input node id out of range");
        }
    }

    for (uint32_t output_idx : sg.graph_outputs) {
        if (dense_legacy_indices) {
            if (output_idx >= sg.nodes.size()) {
                throw std::runtime_error("Graph file corrupted: graph output index out of range");
            }
        } else if (node_ids.find(output_idx) == node_ids.end()) {
            throw std::runtime_error("Graph file corrupted: graph output node id out of range");
        }
    }

    return sg;
}

void save_node(CactusGraph& graph, size_t node_id, const std::string& filename) {
    graph.execute();
    void* data = graph.get_output(node_id);

    const auto& buffer = graph.get_output_buffer(node_id);
    const auto& shape = buffer.shape;
    Precision precision = buffer.precision;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    size_t total_elements = 1;
    for (size_t dim : shape) {
        total_elements *= dim;
    }

    size_t byte_size = PrecisionTraits::packed_size_of(precision, total_elements);

    size_t N = shape.size() >= 1 ? shape[0] : 1;

    uint32_t ndim = static_cast<uint32_t>(shape.size());
    uint32_t flags = 0;
    uint32_t alignment = 32;

    uint32_t magic = CACTUS_MAGIC;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    file.write(reinterpret_cast<const char*>(&alignment), sizeof(alignment));
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

    for (uint32_t i = 0; i < 4; i++) {
        uint64_t dim_val = (i < shape.size()) ? static_cast<uint64_t>(shape[i]) : 0;
        file.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
    }

    uint32_t prec_val = static_cast<uint32_t>(precision);
    file.write(reinterpret_cast<const char*>(&prec_val), sizeof(prec_val));

    uint64_t data_bytes = static_cast<uint64_t>(byte_size);
    uint64_t zero64 = 0;
    uint32_t zero32 = 0;
    file.write(reinterpret_cast<const char*>(&data_bytes), sizeof(data_bytes));
    file.write(reinterpret_cast<const char*>(&zero64), sizeof(zero64));   // scales_bytes = 0
    file.write(reinterpret_cast<const char*>(&zero32), sizeof(zero32));   // group_size = 0
    file.write(reinterpret_cast<const char*>(&zero32), sizeof(zero32));   // num_groups = 0
    uint64_t original_N_val = static_cast<uint64_t>(N);
    file.write(reinterpret_cast<const char*>(&original_N_val), sizeof(original_N_val));

    size_t header_end = HEADER_SIZE;
    size_t aligned_header = align_offset(header_end, alignment);
    size_t header_padding = aligned_header - header_end;
    for (size_t i = 0; i < header_padding; i++) {
        char zero = 0;
        file.write(&zero, 1);
    }

    file.write(static_cast<const char*>(data), byte_size);

    if (!file) {
        throw std::runtime_error("Error writing node data to file: " + filename);
    }
}

// MappedFile implementation

MappedFile::MappedFile(const std::string& filename)
    : fd_(-1), mapped_data_(nullptr), file_size_(0), data_offset_(0) {
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Cannot open file for mapping: " + filename);
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        throw std::runtime_error("Cannot get file size: " + filename);
    }
    file_size_ = static_cast<size_t>(st.st_size);

    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Cannot map file: " + filename);
    }

    close(fd_);
    fd_ = -1;

    parse_header();
    apply_madvise_hints();
}

MappedFile::~MappedFile() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        madvise(mapped_data_, file_size_, MADV_DONTNEED);
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
      data_offset_(other.data_offset_), shape_(std::move(other.shape_)),
      precision_(other.precision_), byte_size_(other.byte_size_),
      group_size_(other.group_size_), num_groups_(other.num_groups_),
      scales_offset_(other.scales_offset_), scales_bytes_(other.scales_bytes_),
      alignment_(other.alignment_),
      is_orthogonal_rotation_(other.is_orthogonal_rotation_),
      is_interleaved_4row_(other.is_interleaved_4row_),
      original_N_(other.original_N_) {
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.is_orthogonal_rotation_ = false;
    other.is_interleaved_4row_ = false;
    other.original_N_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }

        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        shape_ = std::move(other.shape_);
        precision_ = other.precision_;
        byte_size_ = other.byte_size_;
        group_size_ = other.group_size_;
        num_groups_ = other.num_groups_;
        scales_offset_ = other.scales_offset_;
        scales_bytes_ = other.scales_bytes_;
        alignment_ = other.alignment_;
        is_orthogonal_rotation_ = other.is_orthogonal_rotation_;
        is_interleaved_4row_ = other.is_interleaved_4row_;
        original_N_ = other.original_N_;
        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.is_orthogonal_rotation_ = false;
        other.is_interleaved_4row_ = false;
        other.original_N_ = 0;
    }
    return *this;
}

const std::vector<size_t>& MappedFile::shape() const {
    return shape_;
}

Precision MappedFile::precision() const {
    return precision_;
}

size_t MappedFile::byte_size() const {
    return byte_size_;
}

const void* MappedFile::scales_data() const {
    return static_cast<const char*>(mapped_data_) + scales_offset_;
}

void* MappedFile::data() {
    return static_cast<char*>(mapped_data_) + data_offset_;
}

const void* MappedFile::data() const {
    return static_cast<const char*>(mapped_data_) + data_offset_;
}

template<typename T>
const T* MappedFile::typed_data() const {
    return static_cast<const T*>(data());
}

void MappedFile::parse_header() {
    if (file_size_ < HEADER_SIZE) {
        throw std::runtime_error("File too small: insufficient data for header");
    }

    const char* ptr = static_cast<const char*>(mapped_data_);
    size_t offset = 0;

    uint32_t magic = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (magic != CACTUS_MAGIC) {
        throw std::runtime_error("Invalid tensor file: missing CACT magic number");
    }

    uint32_t flags = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    is_orthogonal_rotation_ = (flags & FLAG_ORTHOGONAL_ROTATION) != 0;
    is_interleaved_4row_ = (flags & FLAG_INTERLEAVED_4ROW) != 0;

    alignment_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    if (alignment_ == 0) alignment_ = 1;

    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    std::vector<uint64_t> dims;
    dims.reserve(8);
    for (uint32_t i = 0; i < 4; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        offset += sizeof(uint64_t);
        dims.push_back(dim_val);
    }

    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    precision_ = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);

    byte_size_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    scales_bytes_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    group_size_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    num_groups_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);

    original_N_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
    offset += sizeof(uint64_t);

    size_t header_size = HEADER_SIZE;
    if ((flags & FLAG_EXTENDED_SHAPE) != 0) {
        header_size += 4 * sizeof(uint64_t);
        if (file_size_ < header_size) {
            throw std::runtime_error("File too small: insufficient data for extended shape header");
        }
        for (uint32_t i = 0; i < 4; i++) {
            uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
            offset += sizeof(uint64_t);
            dims.push_back(dim_val);
        }
    }

    shape_.clear();
    if (ndim > dims.size()) {
        throw std::runtime_error("Invalid tensor file: ndim exceeds encoded shape rank");
    }
    for (uint32_t i = 0; i < ndim; i++) {
        if (dims[i] > 0) {
            shape_.push_back(static_cast<size_t>(dims[i]));
        }
    }

    size_t aligned_header = align_offset(header_size, alignment_);

    if (scales_bytes_ > 0) {
        scales_offset_ = aligned_header;
        size_t scales_end = scales_offset_ + scales_bytes_;
        data_offset_ = align_offset(scales_end, alignment_);
    } else {
        scales_offset_ = 0;
        data_offset_ = aligned_header;
    }

    if (data_offset_ + byte_size_ > file_size_) {
        throw std::runtime_error("File corrupted: data extends beyond file size");
    }

}

void MappedFile::apply_madvise_hints() {
    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_WILLNEED);
    }

    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_SEQUENTIAL);
}

void MappedFile::release_pages() {
    if (mapped_data_ == nullptr || mapped_data_ == MAP_FAILED) return;

    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_DONTNEED);
    }
    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_DONTNEED);
}

void MappedFile::prefetch_pages() {
    if (mapped_data_ == nullptr || mapped_data_ == MAP_FAILED) return;

    if (scales_bytes_ > 0 && scales_offset_ > 0) {
        madvise(static_cast<char*>(mapped_data_) + scales_offset_, scales_bytes_, MADV_WILLNEED);
    }
    madvise(static_cast<char*>(mapped_data_) + data_offset_, byte_size_, MADV_WILLNEED);
}

template const int8_t* MappedFile::typed_data<int8_t>() const;
template const float* MappedFile::typed_data<float>() const;
template const uint16_t* MappedFile::typed_data<uint16_t>() const;
template const uint8_t* MappedFile::typed_data<uint8_t>() const;

} // namespace GraphFile
