#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef LM_GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LM_GGML_BACKEND_BUILD
#            define LM_GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define LM_GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define LM_GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define LM_GGML_BACKEND_API extern
#endif

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct lm_ggml_backend_buffer_type * lm_ggml_backend_buffer_type_t;
    typedef struct lm_ggml_backend_buffer * lm_ggml_backend_buffer_t;
    typedef struct lm_ggml_backend_event * lm_ggml_backend_event_t;
    typedef struct lm_ggml_backend * lm_ggml_backend_t;
    typedef void * lm_ggml_backend_graph_plan_t;
    typedef struct lm_ggml_backend_reg * lm_ggml_backend_reg_t;
    typedef struct lm_ggml_backend_device * lm_ggml_backend_dev_t;


    //
    // Backend buffer type
    //

    LM_GGML_API const char *          lm_ggml_backend_buft_name          (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_buft_alloc_buffer  (lm_ggml_backend_buffer_type_t buft, size_t size);
    LM_GGML_API size_t                lm_ggml_backend_buft_get_alignment (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API size_t                lm_ggml_backend_buft_get_max_size  (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API size_t                lm_ggml_backend_buft_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool                  lm_ggml_backend_buft_is_host       (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API lm_ggml_backend_dev_t    lm_ggml_backend_buft_get_device    (lm_ggml_backend_buffer_type_t buft);

    //
    // Backend buffer
    //

    enum lm_ggml_backend_buffer_usage {
        LM_GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
        LM_GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
    };

    LM_GGML_API const char *                   lm_ggml_backend_buffer_name          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void                           lm_ggml_backend_buffer_free          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void *                         lm_ggml_backend_buffer_get_base      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t                         lm_ggml_backend_buffer_get_size      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API enum lm_ggml_status               lm_ggml_backend_buffer_init_tensor   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t                         lm_ggml_backend_buffer_get_alignment (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t                         lm_ggml_backend_buffer_get_max_size  (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t                         lm_ggml_backend_buffer_get_alloc_size(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor);
    LM_GGML_API void                           lm_ggml_backend_buffer_clear         (lm_ggml_backend_buffer_t buffer, uint8_t value);
    LM_GGML_API bool                           lm_ggml_backend_buffer_is_host       (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void                           lm_ggml_backend_buffer_set_usage     (lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage);
    LM_GGML_API enum lm_ggml_backend_buffer_usage lm_ggml_backend_buffer_get_usage     (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API lm_ggml_backend_buffer_type_t     lm_ggml_backend_buffer_get_type      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void                           lm_ggml_backend_buffer_reset         (lm_ggml_backend_buffer_t buffer);

    // tensor copy between different backends
    LM_GGML_API void lm_ggml_backend_tensor_copy(struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    //
    // Backend (stream)
    //

    LM_GGML_API lm_ggml_guid_t  lm_ggml_backend_guid(lm_ggml_backend_t backend);
    LM_GGML_API const char * lm_ggml_backend_name(lm_ggml_backend_t backend);
    LM_GGML_API void         lm_ggml_backend_free(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_get_default_buffer_type(lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_buffer_t      lm_ggml_backend_alloc_buffer(lm_ggml_backend_t backend, size_t size);
    LM_GGML_API size_t                     lm_ggml_backend_get_alignment(lm_ggml_backend_t backend);
    LM_GGML_API size_t                     lm_ggml_backend_get_max_size(lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_tensor_set_async(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get_async(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    // "offset" refers to the offset in tensor->data for setting/getting data
    LM_GGML_API void lm_ggml_backend_tensor_set(      struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get(const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_memset(   struct lm_ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);

    LM_GGML_API void lm_ggml_backend_synchronize(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_graph_plan_t lm_ggml_backend_graph_plan_create(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API void                      lm_ggml_backend_graph_plan_free  (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_plan_compute (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_compute      (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_compute_async(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

    // NOTE: will be removed, use device version instead
    LM_GGML_API bool lm_ggml_backend_supports_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);
    LM_GGML_API bool lm_ggml_backend_supports_buft(lm_ggml_backend_t backend, lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API bool lm_ggml_backend_offload_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    LM_GGML_API void lm_ggml_backend_tensor_copy_async(lm_ggml_backend_t backend_src, lm_ggml_backend_t backend_dst, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_get_device(lm_ggml_backend_t backend);

    //
    // Events
    //

    LM_GGML_API lm_ggml_backend_event_t lm_ggml_backend_event_new(lm_ggml_backend_dev_t device);
    LM_GGML_API void                 lm_ggml_backend_event_free(lm_ggml_backend_event_t event);
    LM_GGML_API void                 lm_ggml_backend_event_record(lm_ggml_backend_event_t event, lm_ggml_backend_t backend);
    LM_GGML_API void                 lm_ggml_backend_event_synchronize(lm_ggml_backend_event_t event);
    LM_GGML_API void                 lm_ggml_backend_event_wait(lm_ggml_backend_t backend, lm_ggml_backend_event_t event);

    //
    // Backend device
    //

    enum lm_ggml_backend_dev_type {
        // CPU device using system memory
        LM_GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        LM_GGML_BACKEND_DEVICE_TYPE_GPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        LM_GGML_BACKEND_DEVICE_TYPE_ACCEL
    };

    // functionality supported by the device
    struct lm_ggml_backend_dev_caps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    // all the device properties
    struct lm_ggml_backend_dev_props {
        const char * name;
        const char * description;
        size_t memory_free;
        size_t memory_total;
        enum lm_ggml_backend_dev_type type;
        struct lm_ggml_backend_dev_caps caps;
    };

    LM_GGML_API const char *                  lm_ggml_backend_dev_name(lm_ggml_backend_dev_t device);
    LM_GGML_API const char *                  lm_ggml_backend_dev_description(lm_ggml_backend_dev_t device);
    LM_GGML_API void                          lm_ggml_backend_dev_memory(lm_ggml_backend_dev_t device, size_t * free, size_t * total);
    LM_GGML_API enum lm_ggml_backend_dev_type    lm_ggml_backend_dev_type(lm_ggml_backend_dev_t device);
    LM_GGML_API void                          lm_ggml_backend_dev_get_props(lm_ggml_backend_dev_t device, struct lm_ggml_backend_dev_props * props);
    LM_GGML_API lm_ggml_backend_reg_t            lm_ggml_backend_dev_backend_reg(lm_ggml_backend_dev_t device);
    LM_GGML_API lm_ggml_backend_t                lm_ggml_backend_dev_init(lm_ggml_backend_dev_t device, const char * params);
    LM_GGML_API lm_ggml_backend_buffer_type_t    lm_ggml_backend_dev_buffer_type(lm_ggml_backend_dev_t device);
    LM_GGML_API lm_ggml_backend_buffer_type_t    lm_ggml_backend_dev_host_buffer_type(lm_ggml_backend_dev_t device);
    LM_GGML_API lm_ggml_backend_buffer_t         lm_ggml_backend_dev_buffer_from_host_ptr(lm_ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);

    LM_GGML_API bool                          lm_ggml_backend_dev_supports_op(lm_ggml_backend_dev_t device, const struct lm_ggml_tensor * op);
    LM_GGML_API bool                          lm_ggml_backend_dev_supports_buft(lm_ggml_backend_dev_t device, lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API bool                          lm_ggml_backend_dev_offload_op(lm_ggml_backend_dev_t device, const struct lm_ggml_tensor * op);

    //
    // Backend (reg)
    //

    LM_GGML_API const char *       lm_ggml_backend_reg_name(lm_ggml_backend_reg_t reg);
    LM_GGML_API size_t             lm_ggml_backend_reg_dev_count(lm_ggml_backend_reg_t reg);
    LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_reg_dev_get(lm_ggml_backend_reg_t reg, size_t index);
    LM_GGML_API void *             lm_ggml_backend_reg_get_proc_address(lm_ggml_backend_reg_t reg, const char * name);

    // Common functions that may be obtained using lm_ggml_backend_reg_get_proc_address

    // Split buffer type for tensor parallelism
    typedef lm_ggml_backend_buffer_type_t   (*lm_ggml_backend_split_buffer_type_t)(int main_device, const float * tensor_split);
    // Set the number of threads for the backend
    typedef void                         (*lm_ggml_backend_set_n_threads_t)(lm_ggml_backend_t backend, int n_threads);
    // Get additional buffer types provided by the device (returns a NULL-terminated array)
    typedef lm_ggml_backend_buffer_type_t * (*lm_ggml_backend_dev_get_extra_bufts_t)(lm_ggml_backend_dev_t device);
    // Set the abort callback for the backend
    typedef void                         (*lm_ggml_backend_set_abort_callback_t)(lm_ggml_backend_t backend, lm_ggml_abort_callback abort_callback, void * abort_callback_data);
    // Get a list of feature flags supported by the backend (returns a NULL-terminated array)
    struct lm_ggml_backend_feature {
        const char * name;
        const char * value;
    };
    typedef struct lm_ggml_backend_feature * (*lm_ggml_backend_get_features_t)(lm_ggml_backend_reg_t reg);

    //
    // Backend registry
    //

    LM_GGML_API void lm_ggml_backend_device_register(lm_ggml_backend_dev_t device);

    // Backend (reg) enumeration
    LM_GGML_API size_t             lm_ggml_backend_reg_count(void);
    LM_GGML_API lm_ggml_backend_reg_t lm_ggml_backend_reg_get(size_t index);
    LM_GGML_API lm_ggml_backend_reg_t lm_ggml_backend_reg_by_name(const char * name);

    // Device enumeration
    LM_GGML_API size_t             lm_ggml_backend_dev_count(void);
    LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_dev_get(size_t index);
    LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_dev_by_name(const char * name);
    LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_dev_by_type(enum lm_ggml_backend_dev_type type);

    // Direct backend (stream) initialization
    // = lm_ggml_backend_dev_init(lm_ggml_backend_dev_by_name(name), params)
    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_init_by_name(const char * name, const char * params);
    // = lm_ggml_backend_dev_init(lm_ggml_backend_dev_by_type(type), params)
    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_init_by_type(enum lm_ggml_backend_dev_type type, const char * params);
    // = lm_ggml_backend_dev_init(lm_ggml_backend_dev_by_type(GPU) OR lm_ggml_backend_dev_by_type(CPU), NULL)
    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_init_best(void);

    // Load a backend from a dynamic library and register it
    LM_GGML_API lm_ggml_backend_reg_t lm_ggml_backend_load(const char * path);
    // Unload a backend if loaded dynamically and unregister it
    LM_GGML_API void               lm_ggml_backend_unload(lm_ggml_backend_reg_t reg);
    // Load all known backends from dynamic libraries
    LM_GGML_API void               lm_ggml_backend_load_all(void);
    LM_GGML_API void               lm_ggml_backend_load_all_from_path(const char * dir_path);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backend devices to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be assigned
        // preferrably to run on the same backend as the buffer
        lm_ggml_backend_buffer_set_usage(buf_weights, LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = lm_ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, LM_GGML_DEFAULT_GRAPH_SIZE, false, true);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct lm_ggml_tensor * node = lm_ggml_mul_mat(ctx, ...);
        lm_ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        lm_ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched); // the graph and its tensors are single-use in terms of allocation, multi-use in terms of computation
        for (int i = 0; i < 10; ++i) {
            lm_ggml_backend_sched_graph_compute(sched, graph); // on the first iteration the graph is allocated automatically
        }

        // if there are graph inputs:
        graph = build_graph(sched); // get a new graph that is not allocated (the metadata for the old graph is freed once lm_ggml_free is called)
        lm_ggml_backend_sched_reset(sched); // clear the allocation of the previous graph
        lm_ggml_backend_sched_alloc_graph(sched, graph); // explicitly allocate the new graph but do not execute it
        lm_ggml_backend_tensor_set(input_tensor, ...); // copy data to the newly allocated graph tensors
        lm_ggml_backend_sched_graph_compute(sched, graph); // execute the graph

        // as an alternative to the above it is also possible to assign the inputs to a dedicated context and
        // allocate them statically via lm_ggml_backend_alloc_ctx_tensors
    }
    */

    typedef struct lm_ggml_backend_sched * lm_ggml_backend_sched_t;

    // Evaluation callback for each node in the graph (set with lm_ggml_backend_sched_set_eval_callback)
    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*lm_ggml_backend_sched_eval_callback)(struct lm_ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler, backends with low index are given priority over backends with high index
    LM_GGML_API lm_ggml_backend_sched_t lm_ggml_backend_sched_new(lm_ggml_backend_t * backends, lm_ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
    LM_GGML_API void                 lm_ggml_backend_sched_free(lm_ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    LM_GGML_API bool                 lm_ggml_backend_sched_reserve(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * measure_graph); // returns success

    LM_GGML_API int                  lm_ggml_backend_sched_get_n_backends(lm_ggml_backend_sched_t sched);
    LM_GGML_API lm_ggml_backend_t       lm_ggml_backend_sched_get_backend(lm_ggml_backend_sched_t sched, int i);

    // Get the number of splits of the last graph
    LM_GGML_API int                  lm_ggml_backend_sched_get_n_splits(lm_ggml_backend_sched_t sched);
    LM_GGML_API int                  lm_ggml_backend_sched_get_n_copies(lm_ggml_backend_sched_t sched);

    LM_GGML_API size_t               lm_ggml_backend_sched_get_buffer_size(lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend);

    LM_GGML_API void                 lm_ggml_backend_sched_set_tensor_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node, lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_t       lm_ggml_backend_sched_get_tensor_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node);

    // Allocate and compute graph on the backend scheduler
    LM_GGML_API bool                 lm_ggml_backend_sched_alloc_graph(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph); // returns success
    LM_GGML_API enum lm_ggml_status     lm_ggml_backend_sched_graph_compute(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph);
    LM_GGML_API enum lm_ggml_status     lm_ggml_backend_sched_graph_compute_async(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph);
    LM_GGML_API void                 lm_ggml_backend_sched_synchronize(lm_ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends or allocating a new graph.
    // This in effect deallocates all tensors that were previously allocated and leaves them with dangling pointers.
    // The correct way to use this API is to discard the deallocated tensors and create new ones.
    LM_GGML_API void                 lm_ggml_backend_sched_reset(lm_ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    LM_GGML_API void                 lm_ggml_backend_sched_set_eval_callback(lm_ggml_backend_sched_t sched, lm_ggml_backend_sched_eval_callback callback, void * user_data);

    //
    // Utils
    //

    struct lm_ggml_backend_graph_copy {
        lm_ggml_backend_buffer_t buffer;
        struct lm_ggml_context * ctx_allocated;
        struct lm_ggml_context * ctx_unallocated;
        struct lm_ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    LM_GGML_API struct lm_ggml_backend_graph_copy lm_ggml_backend_graph_copy(lm_ggml_backend_t backend, struct lm_ggml_cgraph * graph);
    LM_GGML_API void                           lm_ggml_backend_graph_copy_free(struct lm_ggml_backend_graph_copy copy);

    typedef bool (*lm_ggml_backend_eval_callback)(int node_index, struct lm_ggml_tensor * t1, struct lm_ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    LM_GGML_API bool lm_ggml_backend_compare_graph_backend(lm_ggml_backend_t backend1, lm_ggml_backend_t backend2, struct lm_ggml_cgraph * graph, lm_ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_tensor_alloc(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, void * addr);
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_view_init(struct lm_ggml_tensor * tensor);

    // CPU buffer types are always available
    LM_GGML_API lm_ggml_backend_buffer_t      lm_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_buffer_type(void);

#ifdef  __cplusplus
}
#endif
