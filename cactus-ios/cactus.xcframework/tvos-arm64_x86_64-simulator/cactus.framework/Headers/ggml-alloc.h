#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct lm_ggml_backend_buffer_type * lm_ggml_backend_buffer_type_t;
typedef struct      lm_ggml_backend_buffer * lm_ggml_backend_buffer_t;
typedef struct             lm_ggml_backend * lm_ggml_backend_t;

// Tensor allocator
struct lm_ggml_tallocr {
    lm_ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

LM_GGML_API struct lm_ggml_tallocr lm_ggml_tallocr_new(lm_ggml_backend_buffer_t buffer);
LM_GGML_API enum lm_ggml_status    lm_ggml_tallocr_alloc(struct lm_ggml_tallocr * talloc, struct lm_ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    lm_ggml_gallocr_t galloc = lm_ggml_gallocr_new(lm_ggml_backend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    lm_ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct lm_ggml_cgraph * graph = build_graph(batch);
    lm_ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", lm_ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    lm_ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   lm_ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   lm_ggml_set_output(): output tensors are never freed and never overwritten

typedef struct lm_ggml_gallocr * lm_ggml_gallocr_t;

LM_GGML_API lm_ggml_gallocr_t lm_ggml_gallocr_new(lm_ggml_backend_buffer_type_t buft);
LM_GGML_API lm_ggml_gallocr_t lm_ggml_gallocr_new_n(lm_ggml_backend_buffer_type_t * bufts, int n_bufs);
LM_GGML_API void           lm_ggml_gallocr_free(lm_ggml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: lm_ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
LM_GGML_API bool lm_ggml_gallocr_reserve(lm_ggml_gallocr_t galloc, struct lm_ggml_cgraph * graph);
LM_GGML_API bool lm_ggml_gallocr_reserve_n(
    lm_ggml_gallocr_t galloc,
    struct lm_ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call lm_ggml_gallocr_reserve_n first to set the node buffers)
LM_GGML_API bool lm_ggml_gallocr_alloc_graph(lm_ggml_gallocr_t galloc, struct lm_ggml_cgraph * graph);

LM_GGML_API size_t lm_ggml_gallocr_get_buffer_size(lm_ggml_gallocr_t galloc, int buffer_id);

// Utils
// Create a buffer and allocate all the tensors in a lm_ggml_context
LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_backend_alloc_ctx_tensors_from_buft(struct lm_ggml_context * ctx, lm_ggml_backend_buffer_type_t buft);
LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_backend_alloc_ctx_tensors(struct lm_ggml_context * ctx, lm_ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
