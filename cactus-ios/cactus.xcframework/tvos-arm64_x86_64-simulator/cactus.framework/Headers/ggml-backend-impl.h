#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    #define LM_GGML_BACKEND_API_VERSION 1

    //
    // Backend buffer type
    //

    struct lm_ggml_backend_buffer_type_i {
        const char *          (*get_name)      (lm_ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        lm_ggml_backend_buffer_t (*alloc_buffer)  (lm_ggml_backend_buffer_type_t buft, size_t size);
        // tensor alignment
        size_t                (*get_alignment) (lm_ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)  (lm_ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to lm_ggml_nbytes)
        size_t                (*get_alloc_size)(lm_ggml_backend_buffer_type_t buft, const struct lm_ggml_tensor * tensor);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)       (lm_ggml_backend_buffer_type_t buft);
    };

    struct lm_ggml_backend_buffer_type {
        struct lm_ggml_backend_buffer_type_i  iface;
        lm_ggml_backend_dev_t device;
        void * context;
    };

    //
    // Backend buffer
    //

    struct lm_ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (lm_ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (lm_ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum lm_ggml_status (*init_tensor)(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(lm_ggml_backend_buffer_t buffer,       struct lm_ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (lm_ggml_backend_buffer_t buffer,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (lm_ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (lm_ggml_backend_buffer_t buffer);
    };

    struct lm_ggml_backend_buffer {
        struct lm_ggml_backend_buffer_i  iface;
        lm_ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum lm_ggml_backend_buffer_usage usage;
    };

    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_buffer_init(
                   lm_ggml_backend_buffer_type_t buft,
            struct lm_ggml_backend_buffer_i      iface,
                   void *                     context,
                   size_t                     size);

    // do not use directly, use lm_ggml_backend_tensor_copy instead
    LM_GGML_API bool lm_ggml_backend_buffer_copy_tensor(const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_multi_buffer_alloc_buffer(lm_ggml_backend_buffer_t * buffers, size_t n_buffers);
    LM_GGML_API bool                  lm_ggml_backend_buffer_is_multi_buffer(lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void                  lm_ggml_backend_multi_buffer_set_usage(lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage);

    //
    // Backend (stream)
    //

    struct lm_ggml_backend_i {
        const char * (*get_name)(lm_ggml_backend_t backend);

        void (*free)(lm_ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(lm_ggml_backend_t backend_src, lm_ggml_backend_t backend_dst, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(lm_ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        lm_ggml_backend_graph_plan_t (*graph_plan_create) (lm_ggml_backend_t backend, const struct lm_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan, const struct lm_ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum lm_ggml_status          (*graph_plan_compute)(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum lm_ggml_status          (*graph_compute)     (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(lm_ggml_backend_t backend, lm_ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (lm_ggml_backend_t backend, lm_ggml_backend_event_t event);
    };

    struct lm_ggml_backend {
        lm_ggml_guid_t guid;
        struct lm_ggml_backend_i iface;
        lm_ggml_backend_dev_t device;
        void * context;
    };

    struct lm_ggml_backend_event {
        struct lm_ggml_backend_device * device;
        void * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    struct lm_ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(lm_ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(lm_ggml_backend_dev_t dev);

        // device memory in bytes
        void         (*get_memory)(lm_ggml_backend_dev_t dev, size_t * free, size_t * total);

        // device type
        enum lm_ggml_backend_dev_type (*get_type)(lm_ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(lm_ggml_backend_dev_t dev, struct lm_ggml_backend_dev_props * props);

        // backend (stream) initialization
        lm_ggml_backend_t (*init_backend)(lm_ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        lm_ggml_backend_buffer_type_t (*get_buffer_type)(lm_ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        lm_ggml_backend_buffer_type_t (*get_host_buffer_type)(lm_ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        lm_ggml_backend_buffer_t (*buffer_from_host_ptr)(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op);

        // (optional) event synchronization
        lm_ggml_backend_event_t (*event_new)         (lm_ggml_backend_dev_t dev);
        void                 (*event_free)        (lm_ggml_backend_dev_t dev, lm_ggml_backend_event_t event);
        void                 (*event_synchronize) (lm_ggml_backend_dev_t dev, lm_ggml_backend_event_t event);
    };

    struct lm_ggml_backend_device {
        struct lm_ggml_backend_device_i iface;
        lm_ggml_backend_reg_t reg;
        void * context;
    };

    //
    // Backend (reg)
    //

    struct lm_ggml_backend_reg_i {
        const char * (*get_name)(lm_ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_count)(lm_ggml_backend_reg_t reg);
        lm_ggml_backend_dev_t (*get_device)(lm_ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(lm_ggml_backend_reg_t reg, const char * name);
    };

    struct lm_ggml_backend_reg {
        int api_version; // initialize to LM_GGML_BACKEND_API_VERSION
        struct lm_ggml_backend_reg_i iface;
        void * context;
    };

    // Internal backend registry API
    LM_GGML_API void lm_ggml_backend_register(lm_ggml_backend_reg_t reg);

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef lm_ggml_backend_reg_t (*lm_ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*lm_ggml_backend_score_t)(void);

#ifdef LM_GGML_BACKEND_DL
#    ifdef __cplusplus
#        define LM_GGML_BACKEND_DL_IMPL(reg_fn)                             \
            extern "C" {                                                 \
            LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_init(void); \
            }                                                            \
            lm_ggml_backend_reg_t lm_ggml_backend_init(void) {                 \
                return reg_fn();                                         \
            }
#        define LM_GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            extern "C" {                                   \
            LM_GGML_BACKEND_API int lm_ggml_backend_score(void); \
            }                                              \
            int lm_ggml_backend_score(void) {                 \
                return score_fn();                         \
            }
#    else
#        define LM_GGML_BACKEND_DL_IMPL(reg_fn)                              \
            LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_init(void);  \
            lm_ggml_backend_reg_t                  lm_ggml_backend_init(void) { \
                return reg_fn();                                          \
            }
#        define LM_GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            LM_GGML_BACKEND_API int lm_ggml_backend_score(void);  \
            int                  lm_ggml_backend_score(void) { \
                return score_fn();                          \
            }
#    endif
#else
#    define LM_GGML_BACKEND_DL_IMPL(reg_fn)
#    define LM_GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
