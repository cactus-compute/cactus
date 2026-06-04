#include "cactus_ffi.h"
#include "cactus_utils.h"
#include "../graph/graph.h"
#include <vector>
#include <unordered_map>

using namespace cactus::ffi;

struct GraphHandle {
    CactusGraph graph;
};

static GraphHandle* as_graph(cactus_graph_t g) {
    return reinterpret_cast<GraphHandle*>(g);
}

static int fail_invalid(const char* msg) {
    last_error_message = msg;
    return -1;
}

#define GRAPH_WRAP_UNARY(fn_name, method_name) \
int fn_name(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) { \
    if (!graph || !out) return fail_invalid("Invalid args to " #fn_name); \
    try { \
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.method_name(static_cast<size_t>(x))); \
        return 0; \
    } catch (const std::exception& e) { \
        last_error_message = e.what(); \
        return -1; \
    } \
}

#define GRAPH_WRAP_BINARY(fn_name, method_name) \
int fn_name(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) { \
    if (!graph || !out) return fail_invalid("Invalid args to " #fn_name); \
    try { \
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.method_name(static_cast<size_t>(a), static_cast<size_t>(b))); \
        return 0; \
    } catch (const std::exception& e) { \
        last_error_message = e.what(); \
        return -1; \
    } \
}

extern "C" {

cactus_graph_t cactus_graph_create(void) {
    try {
        return reinterpret_cast<cactus_graph_t>(new GraphHandle());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error creating graph";
        return nullptr;
    }
}

void cactus_graph_destroy(cactus_graph_t graph) {
    delete as_graph(graph);
}

int cactus_graph_hard_reset(cactus_graph_t graph) {
    if (!graph) {
        last_error_message = "Invalid args to cactus_graph_hard_reset";
        return -1;
    }
    try {
        as_graph(graph)->graph.hard_reset();
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_save(cactus_graph_t graph, const char* filename) {
    if (!graph || !filename) {
        last_error_message = "Invalid args to cactus_graph_save";
        return -1;
    }
    try {
        as_graph(graph)->graph.save(filename);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}


cactus_graph_t cactus_graph_load(const char* filename) {
    if (!filename) {
        last_error_message = "Invalid args to cactus_graph_load";
        return nullptr;
    }
    try {
        auto handle = std::make_unique<GraphHandle>();
        handle->graph = CactusGraph::load(std::string(filename));
        return reinterpret_cast<cactus_graph_t>(handle.release());
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error loading graph";
        return nullptr;
    }
}

int cactus_graph_input(cactus_graph_t graph, const size_t* shape, size_t rank, int32_t precision, cactus_node_t* out_node) {
    if (!graph || !shape || rank == 0 || !out_node) {
        last_error_message = "Invalid args to cactus_graph_input";
        return -1;
    }
    try {
        std::vector<size_t> s(shape, shape + rank);
        auto id = as_graph(graph)->graph.input(s, static_cast<Precision>(precision));
        *out_node = static_cast<cactus_node_t>(id);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_set_input(cactus_graph_t graph, cactus_node_t node, const void* data, int32_t precision) {
    if (!graph || !data) {
        last_error_message = "Invalid args to cactus_graph_set_input";
        return -1;
    }
    try {
        as_graph(graph)->graph.set_input(static_cast<size_t>(node), data, static_cast<Precision>(precision));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_set_external_input(cactus_graph_t graph, cactus_node_t node, void* data, int32_t precision) {
    if (!graph || !data) {
        return fail_invalid("Invalid args to cactus_graph_set_external_input");
    }
    try {
        as_graph(graph)->graph.set_external_input(static_cast<size_t>(node), data, static_cast<Precision>(precision));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_precision_cast(cactus_graph_t graph, cactus_node_t input, int32_t target_precision, cactus_node_t* out) {
    if (!graph || !out) {
        return fail_invalid("Invalid args to cactus_graph_precision_cast");
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.precision_cast(static_cast<size_t>(input), static_cast<Precision>(target_precision)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_quantize_activations(cactus_graph_t graph, cactus_node_t input, cactus_node_t* out) {
    if (!graph || !out) {
        return fail_invalid("Invalid args to cactus_graph_quantize_activations");
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.quantize_activations(static_cast<size_t>(input)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_add(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_add";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.add(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_add_clipped(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_add_clipped");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.add_clipped(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_subtract(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_subtract";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.subtract(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_multiply(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_multiply";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.multiply(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_divide(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_divide";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.divide(static_cast<size_t>(a), static_cast<size_t>(b)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_add(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t *out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_scalar_add";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_add(static_cast<size_t>(x), value));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_subtract(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_scalar_subtract");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_subtract(static_cast<size_t>(x), value));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_multiply(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t *out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_scalar_multiply";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_multiply(static_cast<size_t>(x), value));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_divide(cactus_graph_t graph, cactus_node_t x, float value, cactus_node_t *out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_scalar_divide";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_divide(static_cast<size_t>(x), value));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_exp(cactus_graph_t graph, cactus_node_t x, cactus_node_t *out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_scalar_exp";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_exp(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_sqrt(cactus_graph_t graph, cactus_node_t x, cactus_node_t *out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_scalar_sqrt";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_sqrt(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_cos(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_scalar_cos");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_cos(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_sin(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_scalar_sin");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_sin(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scalar_log(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_scalar_log");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scalar_log(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_abs(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_abs";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.abs(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_pow(cactus_graph_t graph, cactus_node_t x, float exponent, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_pow";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.pow(static_cast<size_t>(x), exponent));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_view(cactus_graph_t graph, cactus_node_t x, const size_t* shape, size_t rank, cactus_node_t* out) {
    if (!graph || !shape || rank == 0 || !out) {
        last_error_message = "Invalid args to cactus_graph_view";
        return -1;
    }
    try {
        std::vector<size_t> s(shape, shape + rank);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.view(static_cast<size_t>(x), s));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_flatten(cactus_graph_t graph, cactus_node_t x, int32_t start_dim, int32_t end_dim, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_flatten";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.flatten(static_cast<size_t>(x), start_dim, end_dim));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_reshape(cactus_graph_t graph, cactus_node_t x, const size_t* shape, size_t rank, cactus_node_t* out) {
    if (!graph || !shape || rank == 0 || !out) return fail_invalid("Invalid args to cactus_graph_reshape");
    try {
        std::vector<size_t> s(shape, shape + rank);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.reshape(static_cast<size_t>(x), s));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_transpose(cactus_graph_t graph, cactus_node_t x, int32_t backend, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_transpose");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.transpose(static_cast<size_t>(x), static_cast<ComputeBackend>(backend)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_transpose_n(cactus_graph_t graph, cactus_node_t x, const size_t* permutation, size_t rank, int32_t backend, cactus_node_t* out) {
    if (!graph || !permutation || rank == 0 || !out) return fail_invalid("Invalid args to cactus_graph_transpose_n");
    try {
        std::vector<size_t> p(permutation, permutation + rank);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.transposeN(static_cast<size_t>(x), p, static_cast<ComputeBackend>(backend)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_slice(cactus_graph_t graph, cactus_node_t x, int32_t axis, size_t start, size_t length, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_slice");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.slice(static_cast<size_t>(x), axis, start, length));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_index(cactus_graph_t graph, cactus_node_t x, size_t index_value, int32_t dim, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_index");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.index(static_cast<size_t>(x), index_value, dim));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_sum(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_sum");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.sum(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_mean(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_mean");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.mean(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_variance(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_variance");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.variance(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_min(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_min");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.min(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_max(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_max");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.max(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_concat(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) {
        last_error_message = "Invalid args to cactus_graph_concat";
        return -1;
    }
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.concat(static_cast<size_t>(a), static_cast<size_t>(b), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_cat(cactus_graph_t graph, const cactus_node_t* nodes, size_t count, int32_t axis, cactus_node_t* out) {
    if (!graph || !nodes || !out || count == 0) {
        last_error_message = "Invalid args to cactus_graph_cat";
        return -1;
    }
    try {
        std::vector<size_t> ns;
        for (size_t i = 0; i < count; ++i) {
            ns.push_back(static_cast<size_t>(nodes[i]));
        }
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.cat(ns, axis));
        return 0;

    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_matmul(cactus_graph_t graph, cactus_node_t a, cactus_node_t b, bool pretransposed_rhs, int32_t backend, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_matmul");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.matmul(static_cast<size_t>(a), static_cast<size_t>(b), pretransposed_rhs, static_cast<ComputeBackend>(backend)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gather(cactus_graph_t graph, cactus_node_t tensor, cactus_node_t indices, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gather");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gather(static_cast<size_t>(tensor), static_cast<size_t>(indices)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_embedding_from_tensor(cactus_graph_t graph, cactus_node_t embedding_tensor, cactus_node_t indices, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_embedding_from_tensor");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.embedding(static_cast<size_t>(embedding_tensor), static_cast<size_t>(indices)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_embedding_from_file(cactus_graph_t graph, const char* filename, cactus_node_t indices, cactus_node_t* out) {
    if (!graph || !filename || !out) return fail_invalid("Invalid args to cactus_graph_embedding_from_file");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.embedding(std::string(filename), static_cast<size_t>(indices)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_mmap_embeddings(cactus_graph_t graph, const char* filename, cactus_node_t* out) {
    if (!graph || !filename || !out) return fail_invalid("Invalid args to cactus_graph_mmap_embeddings");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.mmap_embeddings(std::string(filename)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_mmap_weights(cactus_graph_t graph, const char* filename, cactus_node_t* out) {
    if (!graph || !filename || !out) return fail_invalid("Invalid args to cactus_graph_mmap_weights");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.mmap_weights(std::string(filename)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_bilinear_interpolation(cactus_graph_t graph, cactus_node_t pos_embeds, size_t dst_height, size_t dst_width, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_bilinear_interpolation");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.bilinear_interpolation(static_cast<size_t>(pos_embeds), dst_height, dst_width));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_set_grouped_scales(cactus_graph_t graph, cactus_node_t node, size_t group_size, size_t num_groups, void* scales_ptr) {
    if (!graph || !scales_ptr) return fail_invalid("Invalid args to cactus_graph_set_grouped_scales");
    try {
        as_graph(graph)->graph.set_grouped_scales(static_cast<size_t>(node), group_size, num_groups, scales_ptr);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_set_interleaved(cactus_graph_t graph, cactus_node_t node, bool interleaved, size_t original_n) {
    if (!graph) return fail_invalid("Invalid args to cactus_graph_set_interleaved");
    try {
        as_graph(graph)->graph.set_interleaved(static_cast<size_t>(node), interleaved, original_n);
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_release_weight_pages(cactus_graph_t graph, cactus_node_t node) {
    if (!graph) return fail_invalid("Invalid args to cactus_graph_release_weight_pages");
    try {
        as_graph(graph)->graph.release_weight_pages(static_cast<size_t>(node));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_prefetch_weight_pages(cactus_graph_t graph, cactus_node_t node) {
    if (!graph) return fail_invalid("Invalid args to cactus_graph_prefetch_weight_pages");
    try {
        as_graph(graph)->graph.prefetch_weight_pages(static_cast<size_t>(node));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_release_all_weight_pages(cactus_graph_t graph) {
    if (!graph) return fail_invalid("Invalid args to cactus_graph_release_all_weight_pages");
    try {
        as_graph(graph)->graph.release_all_weight_pages();
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_relu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_relu");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.relu(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_silu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_silu");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.silu(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gelu(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gelu");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gelu(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gelu_erf(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gelu_erf");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gelu_erf(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_sigmoid(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_sigmoid");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.sigmoid(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_tanh(cactus_graph_t graph, cactus_node_t x, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_tanh");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.tanh(static_cast<size_t>(x)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_glu(cactus_graph_t graph, cactus_node_t x, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_glu");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.glu(static_cast<size_t>(x), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_layernorm(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, float epsilon, bool has_bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_layernorm");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.layernorm(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias), epsilon));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.layernorm(static_cast<size_t>(input), static_cast<size_t>(weight), epsilon));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_groupnorm(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, size_t num_groups, float epsilon, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_groupnorm");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.groupnorm(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias), num_groups, epsilon));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_batchnorm(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, cactus_node_t running_mean, cactus_node_t running_var, int32_t axis, float epsilon, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_batchnorm");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.batchnorm(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias), static_cast<size_t>(running_mean), static_cast<size_t>(running_var), axis, epsilon));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_topk(cactus_graph_t graph, cactus_node_t input, size_t k, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_topk");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.topk(static_cast<size_t>(input), k));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_rms_norm(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, float epsilon, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_rms_norm");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.rms_norm(static_cast<size_t>(input), static_cast<size_t>(weight), epsilon));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_rope(cactus_graph_t graph, cactus_node_t input, float theta, size_t position_offset, int32_t backend, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_rope");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.rope(static_cast<size_t>(input), theta, position_offset, static_cast<ComputeBackend>(backend)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_rope_gptj(cactus_graph_t graph, cactus_node_t input, float theta, size_t position_offset, size_t rot_dim, int32_t backend, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_rope_gptj");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.rope_gptj(static_cast<size_t>(input), theta, position_offset, rot_dim, static_cast<ComputeBackend>(backend)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_softmax(cactus_graph_t graph, cactus_node_t input, int32_t axis, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_softmax");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.softmax(static_cast<size_t>(input), axis));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_attention(cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, float scale, bool is_causal, size_t position_offset, size_t window_size, int32_t backend, bool use_mask, cactus_node_t mask, bool additive_mask, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_attention");
    try {
        if (use_mask) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.attention_masked(static_cast<size_t>(query), static_cast<size_t>(key), static_cast<size_t>(value), static_cast<size_t>(mask), scale, is_causal, static_cast<ComputeBackend>(backend), additive_mask, position_offset, window_size));
        } else if (window_size > 0 || position_offset > 0) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.attention(static_cast<size_t>(query), static_cast<size_t>(key), static_cast<size_t>(value), scale, position_offset, window_size, static_cast<ComputeBackend>(backend)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.attention(static_cast<size_t>(query), static_cast<size_t>(key), static_cast<size_t>(value), scale, is_causal, static_cast<ComputeBackend>(backend)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_rel_pos_bias(cactus_graph_t graph, cactus_node_t query, cactus_node_t relative_key, float scale, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_rel_pos_bias");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.rel_pos_bias(static_cast<size_t>(query), static_cast<size_t>(relative_key), scale));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_attention_int8_hybrid(cactus_graph_t graph, cactus_node_t query, cactus_node_t key_new, cactus_node_t value_new, float scale, size_t position_offset,
                                       const int8_t* cached_keys, const int8_t* cached_values, const float* k_scales, const float* v_scales,
                                       size_t cache_len, size_t num_kv_heads, size_t head_dim, size_t window_size, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_attention_int8_hybrid");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.attention_int8_hybrid(static_cast<size_t>(query), static_cast<size_t>(key_new), static_cast<size_t>(value_new), scale, position_offset, cached_keys, cached_values, k_scales, v_scales, cache_len, num_kv_heads, head_dim, window_size));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d_causal(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t kernel_size, size_t dilation, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d_causal");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_causal(static_cast<size_t>(input), static_cast<size_t>(weight), kernel_size, dilation));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d_k3(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t stride, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d_k3");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_k3(static_cast<size_t>(input), static_cast<size_t>(weight), stride));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d_k7s3(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d_k7s3");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_k7s3(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, size_t stride, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias), stride));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d(static_cast<size_t>(input), static_cast<size_t>(weight), stride));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d_same_depthwise_k9(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d_same_depthwise_k9");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_same_depthwise_k9(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_same_depthwise_k9(static_cast<size_t>(input), static_cast<size_t>(weight)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv1d_pointwise(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv1d_pointwise");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_pointwise(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv1d_pointwise(static_cast<size_t>(input), static_cast<size_t>(weight)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv2d_k3s2p1(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv2d_k3s2p1");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_k3s2p1(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_k3s2p1(static_cast<size_t>(input), static_cast<size_t>(weight)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv2d_depthwise_k3s2p1(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv2d_depthwise_k3s2p1");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_depthwise_k3s2p1(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_depthwise_k3s2p1(static_cast<size_t>(input), static_cast<size_t>(weight)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_conv2d_pointwise_1x1(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, bool has_bias, cactus_node_t bias, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_conv2d_pointwise_1x1");
    try {
        if (has_bias) {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_pointwise_1x1(static_cast<size_t>(input), static_cast<size_t>(weight), static_cast<size_t>(bias)));
        } else {
            *out = static_cast<cactus_node_t>(as_graph(graph)->graph.conv2d_pointwise_1x1(static_cast<size_t>(input), static_cast<size_t>(weight)));
        }
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_lstm_cell(cactus_graph_t graph, cactus_node_t input, cactus_node_t h_prev, cactus_node_t c_prev, cactus_node_t weight_ih, cactus_node_t weight_hh, cactus_node_t bias_ih, cactus_node_t bias_hh, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_lstm_cell");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.lstm_cell(static_cast<size_t>(input), static_cast<size_t>(h_prev), static_cast<size_t>(c_prev), static_cast<size_t>(weight_ih), static_cast<size_t>(weight_hh), static_cast<size_t>(bias_ih), static_cast<size_t>(bias_hh)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gated_deltanet_decode(cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, cactus_node_t gate_log, cactus_node_t beta, cactus_node_t initial_state, float scale, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gated_deltanet_decode");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gated_deltanet_decode(static_cast<size_t>(query), static_cast<size_t>(key), static_cast<size_t>(value), static_cast<size_t>(gate_log), static_cast<size_t>(beta), static_cast<size_t>(initial_state), scale));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gated_deltanet_prefill(cactus_graph_t graph, cactus_node_t query, cactus_node_t key, cactus_node_t value, cactus_node_t gate_log, cactus_node_t beta, cactus_node_t initial_state, size_t chunk_size, float scale, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gated_deltanet_prefill");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gated_deltanet_prefill(static_cast<size_t>(query), static_cast<size_t>(key), static_cast<size_t>(value), static_cast<size_t>(gate_log), static_cast<size_t>(beta), static_cast<size_t>(initial_state), chunk_size, scale));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_stft(cactus_graph_t graph, cactus_node_t input, cactus_node_t weight, size_t stride, size_t num_fft_bins, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_stft");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.stft(static_cast<size_t>(input), static_cast<size_t>(weight), stride, num_fft_bins));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_altup_predict(cactus_graph_t graph, cactus_node_t coefs, const cactus_node_t* streams, size_t num_streams, cactus_node_t* out) {
    if (!graph || !streams || num_streams == 0 || !out) return fail_invalid("Invalid args to cactus_graph_altup_predict");
    try {
        std::vector<size_t> s(num_streams);
        for (size_t i = 0; i < num_streams; ++i) s[i] = static_cast<size_t>(streams[i]);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.altup_predict(static_cast<size_t>(coefs), s.data(), num_streams));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_altup_correct(cactus_graph_t graph, cactus_node_t coefs, cactus_node_t innovation, const cactus_node_t* predictions, size_t num_predictions, cactus_node_t* out) {
    if (!graph || !predictions || num_predictions == 0 || !out) return fail_invalid("Invalid args to cactus_graph_altup_correct");
    try {
        std::vector<size_t> p(num_predictions);
        for (size_t i = 0; i < num_predictions; ++i) p[i] = static_cast<size_t>(predictions[i]);
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.altup_correct(static_cast<size_t>(coefs), static_cast<size_t>(innovation), p.data(), num_predictions));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_gaussian_topk(cactus_graph_t graph, cactus_node_t input, float ppf, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_gaussian_topk");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.gaussian_topk(static_cast<size_t>(input), ppf));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_moe_layer_gated(cactus_graph_t graph, cactus_node_t hidden, cactus_node_t routing_probs, cactus_node_t topk_indices,
                                 const cactus_node_t* w1_weights, const cactus_node_t* w3_weights, const cactus_node_t* w2_weights,
                                 size_t num_experts, size_t num_experts_per_tok, bool normalize_routing, float epsilon, float routed_scaling_factor, cactus_node_t* out) {
    if (!graph || !w1_weights || !w3_weights || !w2_weights || !out) return fail_invalid("Invalid args to cactus_graph_moe_layer_gated");
    try {
        std::vector<size_t> w1(num_experts), w3(num_experts), w2(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            w1[i] = static_cast<size_t>(w1_weights[i]);
            w3[i] = static_cast<size_t>(w3_weights[i]);
            w2[i] = static_cast<size_t>(w2_weights[i]);
        }
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.moe_layer(static_cast<size_t>(hidden), static_cast<size_t>(routing_probs), static_cast<size_t>(topk_indices), w1, w3, w2, num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_moe_layer_ungated(cactus_graph_t graph, cactus_node_t hidden, cactus_node_t routing_probs, cactus_node_t topk_indices,
                                   const cactus_node_t* w1_weights, const cactus_node_t* w2_weights,
                                   size_t num_experts, size_t num_experts_per_tok, bool normalize_routing, float epsilon, float routed_scaling_factor, int32_t activation, cactus_node_t* out) {
    if (!graph || !w1_weights || !w2_weights || !out) return fail_invalid("Invalid args to cactus_graph_moe_layer_ungated");
    try {
        std::vector<size_t> w1(num_experts), w2(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            w1[i] = static_cast<size_t>(w1_weights[i]);
            w2[i] = static_cast<size_t>(w2_weights[i]);
        }
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.moe_layer(static_cast<size_t>(hidden), static_cast<size_t>(routing_probs), static_cast<size_t>(topk_indices), w1, w2, num_experts, num_experts_per_tok, normalize_routing, epsilon, routed_scaling_factor, static_cast<Activation>(activation)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_sample(cactus_graph_t graph, cactus_node_t logits, float temperature, float top_p, size_t top_k, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_sample");
    try {
        std::unordered_map<uint32_t, float> empty_bias;
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.sample(static_cast<size_t>(logits), temperature, top_p, top_k, empty_bias));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_scatter_topk(cactus_graph_t graph, cactus_node_t indices, cactus_node_t values, size_t num_classes, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_scatter_topk");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.scatter_topk(static_cast<size_t>(indices), static_cast<size_t>(values), num_classes));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_persistent(cactus_graph_t graph, cactus_node_t source_node, cactus_node_t* out) {
    if (!graph || !out) return fail_invalid("Invalid args to cactus_graph_persistent");
    try {
        *out = static_cast<cactus_node_t>(as_graph(graph)->graph.persistent(static_cast<size_t>(source_node)));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_is_populated(cactus_graph_t graph, cactus_node_t persistent_node, int32_t* out_is_populated) {
    if (!graph || !out_is_populated) return fail_invalid("Invalid args to cactus_graph_is_populated");
    try {
        *out_is_populated = as_graph(graph)->graph.is_populated(static_cast<size_t>(persistent_node)) ? 1 : 0;
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_invalidate_persistent(cactus_graph_t graph, cactus_node_t persistent_node) {
    if (!graph) return fail_invalid("Invalid args to cactus_graph_invalidate_persistent");
    try {
        as_graph(graph)->graph.invalidate_persistent(static_cast<size_t>(persistent_node));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_execute(cactus_graph_t graph) {
    if (!graph) {
        last_error_message = "Graph is null";
        return -1;
    }
    try {
        as_graph(graph)->graph.execute();
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_get_output_ptr(cactus_graph_t graph, cactus_node_t node, void** out_ptr) {
    if (!graph || !out_ptr) {
        last_error_message = "Invalid args to cactus_graph_get_output_ptr";
        return -1;
    }
    try {
        *out_ptr = as_graph(graph)->graph.get_output(static_cast<size_t>(node));
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}

int cactus_graph_get_output_info(cactus_graph_t graph, cactus_node_t node, cactus_tensor_info_t* out_info) {
    if (!graph || !out_info) {
        last_error_message = "Invalid args to cactus_graph_get_output_info";
        return -1;
    }
    try {
        const auto& buf = as_graph(graph)->graph.get_output_buffer(static_cast<size_t>(node));
        out_info->precision = static_cast<int32_t>(buf.precision);
        out_info->rank = buf.shape.size();
        if (out_info->rank > 8) {
            last_error_message = "Rank exceeds cactus_tensor_info_t shape capacity";
            return -1;
        }
        for (size_t i = 0; i < out_info->rank; ++i) {
            out_info->shape[i] = buf.shape[i];
        }
        for (size_t i = out_info->rank; i < 8; ++i) {
            out_info->shape[i] = 0;
        }
        out_info->num_elements = buf.total_size;
        out_info->byte_size = buf.byte_size;
        return 0;
    } catch (const std::exception& e) {
        last_error_message = e.what();
        return -1;
    }
}
}
