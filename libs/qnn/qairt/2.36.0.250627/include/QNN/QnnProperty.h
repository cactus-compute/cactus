//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Property component API.
 *
 *          Provides means for client to discover capabilities of a backend.
 */

#ifndef QNN_PROPERTY_H
#define QNN_PROPERTY_H

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
///
/// Definition of QNN_PROPERTY_GROUP_CORE property group.
///

/**
 * @brief Property group for the QNN core property group.
 */
#define QNN_PROPERTY_GROUP_CORE 0x00000001

///
/// Definition of QNN_PROPERTY_GROUP_BACKEND property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Backend API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_BACKEND (QNN_PROPERTY_GROUP_CORE + 100)

/**
 * @brief Property key for determining if a backend supports QnnBackend_registerOpPackage.
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_OP_PACKAGE (QNN_PROPERTY_GROUP_BACKEND + 4)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_BACKEND_CONFIG_OPTION_PLATFORM configuration.
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_PLATFORM_OPTIONS (QNN_PROPERTY_GROUP_BACKEND + 5)

/**
 * @brief Property key for determining whether a backend supports graph composition.
 *        The following are considered graph composition APIs:
 *        - QnnContext_create
 *        - QnnGraph_create
 *        - QnnGraph_addNode
 *        - QnnGraph_finalize
 *        - QnnTensor_createContextTensor
 *        - QnnTensor_createGraphTensor
 *        - QnnBackend_validateOpConfig
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_COMPOSITION (QNN_PROPERTY_GROUP_BACKEND + 6)

/**
 * @brief Property key for determining whether a backend supports setting
 *        QNN_BACKEND_PROPERTY_OPTION_CUSTOM as a property option.
 */
#define QNN_PROPERTY_BACKEND_SUPPORT_CUSTOM_PROPERTY (QNN_PROPERTY_GROUP_BACKEND + 7)

///
/// Definition of QNN_PROPERTY_GROUP_CONTEXT property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Context API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_CONTEXT (QNN_PROPERTY_GROUP_CORE + 200)

/**
 * @brief Property key for determining whether a backend supports context binaries. It determines
 *        supports for the following APIs:
 *        - QnnContext_getBinarySize
 *        - QnnContext_getBinary
 *        - QnnContext_createFromBinary
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CACHING (QNN_PROPERTY_GROUP_CONTEXT + 1)

/**
 * @brief Property key for determining whether a backend supports the QnnContext_Config_t data
 *        structure.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIGURATION (QNN_PROPERTY_GROUP_CONTEXT + 4)

/**
 * @brief Property key for determining whether a backend supports graph enablement in a context. See
 *        QNN_CONTEXT_CONFIG_ENABLE_GRAPHS.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_ENABLE_GRAPHS (QNN_PROPERTY_GROUP_CONTEXT + 5)

/**
 * @brief Property key for determining whether a backend supports memory limits in a context. See
 *        QNN_CONTEXT_CONFIG_MEMORY_LIMIT.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_MEMORY_LIMIT_HINT (QNN_PROPERTY_GROUP_CONTEXT + 6)

/**
 * @brief Property key for determining whether a backend supports context binaries that are readable
 *        throughout the lifetime of the context. See QNN_CONTEXT_CONFIG_PERSISTENT_BINARY.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_PERSISTENT_BINARY (QNN_PROPERTY_GROUP_CONTEXT + 7)

/**
 * @brief Property key for determining whether a backend supports binary compatibility control in a
 *        context. See QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CONFIG_BINARY_COMPATIBILITY_TYPE \
  (QNN_PROPERTY_GROUP_CONTEXT + 8)

/**
 * @brief Property key for determining whether a backend supports validation of a stored binary. It
 *        determines support for QnnContext_validateBinary.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_VALIDATE_BINARY (QNN_PROPERTY_GROUP_CONTEXT + 9)

/**
 * @brief Property key for determining whether a backend supports creating a context from a stored
 *        binary, which supports control signals. It determines support for
 *        QnnContext_createFromBinaryWithSignal.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_WITH_SIGNALS \
  (QNN_PROPERTY_GROUP_CONTEXT + 10)

/**
 * @brief Property key for determining whether a backend supports creating multiple contexts from
 *        binaries in a single API call. It determines support for
 *        QnnContext_createFromBinaryListAsync.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC (QNN_PROPERTY_GROUP_CONTEXT + 11)

/**
 * @brief Property key for determining whether a backend supports creation and application of
 *        updates for an existing context binary. This determines support for
 *        QnnContext_getBinarySectionSize(), QnnContext_retrieveBinarySection(), and
 *        QnnContext_applyBinarySection() with QNN_CONTEXT_SECTION_UPDATABLE.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_UPDATES (QNN_PROPERTY_GROUP_CONTEXT + 12)

/**
 * @brief Property key for determining whether a backend supports use of binary sections without the
 *        __graph__ argument provided.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_SECTION_FULL_CONTEXT (QNN_PROPERTY_GROUP_CONTEXT + 13)

/**
 * @brief Property key for determining whether a backend supports setting
 *        QNN_CONTEXT_PROPERTY_OPTION_CUSTOM as a property option.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_CUSTOM_PROPERTY (QNN_PROPERTY_GROUP_CONTEXT + 14)

/**
 * @brief Property key for determining whether a backend supports QnnContext_getIncrementalBinary
 *        and QnnContext_releaseIncrementalBinary.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_INCREMENTAL_BINARY (QNN_PROPERTY_GROUP_CONTEXT + 15)

/**
 * @brief Property key for determining whether a backend supports deferred graph initialization
 *        during context creation. See QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_DEFERRED_GRAPH_INIT (QNN_PROPERTY_GROUP_CONTEXT + 16)

/**
 * @brief Property key for determining whether a backend supports creation and application of
 *        weight only updates for an existing context binary. This determines support for
 *        QnnContext_getBinarySectionSize(), QnnContext_retrieveBinarySection(), and
 *        QnnContext_applyBinarySection() with QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_WEIGHT_ONLY_UPDATES (QNN_PROPERTY_GROUP_CONTEXT + 17)

/**
 * @brief Property key for determining whether a backend supports creation and application of
 *        quant param only updates for an existing context binary. This determines support for
 *        QnnContext_getBinarySectionSize(), QnnContext_retrieveBinarySection(), and
 *        QnnContext_applyBinarySection() with QNN_CONTEXT_SECTION_UPDATABLE_QUANT_PARAMS.
 */
#define QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_QUANT_ONLY_UPDATES (QNN_PROPERTY_GROUP_CONTEXT + 18)

///
/// Definition of QNN_PROPERTY_GROUP_GRAPH property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Graph API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_GRAPH (QNN_PROPERTY_GROUP_CORE + 300)

/**
 * @brief Property key for determining whether a backend supports graph configuration. It determines
 *        support for QnnGraph_setConfig.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_CONFIG (QNN_PROPERTY_GROUP_GRAPH + 1)

/**
 * @brief Property key for determining whether a backend supports signals.
 * @note This capability is equivalent to all of QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL,
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL, and
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL having support.
 * @note DEPRECATED: Use QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL,
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL, or
 *       QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL for QnnGraph API support for QnnSignal.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_SIGNALS (QNN_PROPERTY_GROUP_GRAPH + 2)

/**
 * @brief Property key for determining whether a backend supports asynchronous graph execution. It
 *        determines support for QnnGraph_executeAsync.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_ASYNC_EXECUTION (QNN_PROPERTY_GROUP_GRAPH + 3)

/**
 * @brief Property key for determining whether a backend supports execution of graphs with null
 *        inputs. This implies that the graph will contain no APP_WRITE tensors.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_NULL_INPUTS (QNN_PROPERTY_GROUP_GRAPH + 4)

/**
 * @brief Property key for determining whether a backend supports priority control of graphs within
 *        a context. See QNN_GRAPH_CONFIG_OPTION_PRIORITY.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_PRIORITY_CONTROL (QNN_PROPERTY_GROUP_GRAPH + 5)

/**
 * @brief Property key for determining whether a backend supports QnnSignal for QnnGraph_finalize.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 6)

/**
 * @brief Property key for determining whether a backend supports QnnSignal for QnnGraph_execute.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 7)

/**
 * @brief Property key for determining whether a backend supports QnnSignal for
 *        QnnGraph_executeAsync.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_ASYNC_SIGNAL (QNN_PROPERTY_GROUP_GRAPH + 8)

/**
 * @brief Property key for determining whether a backend supports graph-level continuous profiling.
 *        See QNN_GRAPH_CONFIG_OPTION_PROFILE_HANDLE.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_CONTINUOUS_PROFILING (QNN_PROPERTY_GROUP_GRAPH + 9)

/**
 * @brief Property key for determining whether a backend supports graph execution. It determines
 *        support for QnnGraph_execute.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE (QNN_PROPERTY_GROUP_GRAPH + 10)

/**
 * @brief Property key for determining whether a backend supports batch multiplier.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE (QNN_PROPERTY_GROUP_GRAPH + 11)

/**
 * @brief Property key for determining whether a backend supports per-API profiling data
 *        for graph execution.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EXECUTE_PER_API_PROFILING (QNN_PROPERTY_GROUP_GRAPH + 12)

/**
 * @brief Property key for determining whether a backend supports subgraphs. It determines support
 *        for QnnGraph_createSubgraph.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_SUBGRAPH (QNN_PROPERTY_GROUP_GRAPH + 13)

/**
 * @brief Property key for determining whether a backend supports graph profiling state. See
 *        QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_STATE.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_PROFILING_STATE (QNN_PROPERTY_GROUP_GRAPH + 14)

/**
 * @brief Property key for determining whether a backend supports controlling the number of
 *        profiling executions of a graph. See QNN_GRAPH_CONFIG_OPTION_SET_PROFILING_NUM_EXECUTIONS.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_SET_PROFILING_NUM_EXECUTIONS (QNN_PROPERTY_GROUP_GRAPH + 15)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_BIND_MEM_HANDLES execution environment option for
 *        binding client allocated mem handles to a graph.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_BIND_MEM_HANDLES (QNN_PROPERTY_GROUP_GRAPH + 16)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_GRAPH_EXECUTE_ENVIRONMENT_OPTION_POPULATE_CLIENT_BUFS execution environment option for
 *        populating client buffers with backend allocated memory.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_ENV_OPTION_POPULATE_CLIENT_BUFS (QNN_PROPERTY_GROUP_GRAPH + 17)

/**
 * @brief Property key for determining whether a backend supports finalizing
 *        (QnnGraph_finalize) a graph retrieved from a context binary.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_FINALIZE_DESERIALIZED_GRAPH (QNN_PROPERTY_GROUP_GRAPH + 18)

/**
 * @brief Property key for determining whether a backend supports setting
 *        QNN_GRAPH_PROPERTY_OPTION_CUSTOM as a property option.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_CUSTOM_PROPERTY (QNN_PROPERTY_GROUP_GRAPH + 19)

/**
 * @brief Property key for determining whether a backend supports early termination of graph
 *        execution.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION (QNN_PROPERTY_GROUP_GRAPH + 20)

/**
 * @brief Property key for determining whether a backend supports online preparation of
 *        graphs.
 */
#define QNN_PROPERTY_GRAPH_SUPPORT_ONLINE_PREPARE (QNN_PROPERTY_GROUP_GRAPH + 21)

///
/// Definition of QNN_PROPERTY_GROUP_OP_PACKAGE property group. This group is Optional portion of
/// API.
///

/**
 * @brief Property group for the QNN Op Package API property group. This can be used as a key to
 *        check if Op Package API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_OP_PACKAGE (QNN_PROPERTY_GROUP_CORE + 400)

/**
 * @brief Property key for determining whether an op package supports validation.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_VALIDATION (QNN_PROPERTY_GROUP_OP_PACKAGE + 1)

/**
 * @brief Property key for determining whether an op package supports op implementation creation and
 *        freeing.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_OP_IMPLS (QNN_PROPERTY_GROUP_OP_PACKAGE + 2)

/**
 * @brief Property key for determining whether an op package supports duplication of operation
 *        names, such that there are duplicated op_package_name::op_name combinations.
 */
#define QNN_PROPERTY_OP_PACKAGE_SUPPORTS_DUPLICATE_NAMES (QNN_PROPERTY_GROUP_OP_PACKAGE + 3)

///
/// Definition of QNN_PROPERTY_GROUP_TENSOR property group. This group is Core (non-optional) API.
///

/**
 * @brief Property group for the QNN Tensor API property group. This is a non-optional API
 *        component and cannot be used as a property key.
 */
#define QNN_PROPERTY_GROUP_TENSOR (QNN_PROPERTY_GROUP_CORE + 500)

/**
 * @brief Property key to determine whether a backend supports Qnn_MemHandle_t type tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_MEMHANDLE_TYPE (QNN_PROPERTY_GROUP_TENSOR + 1)

/**
 * @brief Property key to determine whether a backend supports creating context tensors. It
 *        determines support for QnnTensor_createContextTensor.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_CONTEXT_TENSORS (QNN_PROPERTY_GROUP_TENSOR + 2)

/**
 * @brief Property key to determine whether a backend supports dynamic tensor dimensions.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS (QNN_PROPERTY_GROUP_TENSOR + 3)

/**
 * @brief Property key to determine whether a backend supports tensor sparsity.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY (QNN_PROPERTY_GROUP_TENSOR + 4)

/**
 * @brief Property key to determine whether a backend supports updating static tensor weight data
 *        and quantization encodings, if applicable.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_STATIC_TENSORS (QNN_PROPERTY_GROUP_TENSOR + 5)

/**
 * @brief Property key to determine whether a backend supports updating quantization tensor
 *        encodings for UPDATABLE_NATIVE tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_NATIVE_TENSORS (QNN_PROPERTY_GROUP_TENSOR + 6)

/**
 * @brief Property key to determine whether a backend supports updating quantization tensor
 *        encodings for UPDATABLE_APP_READ, UPDATABLE_APP_WRITE, and UPDATABLE_APP_READWRITE
 *        tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_APP_TENSORS (QNN_PROPERTY_GROUP_TENSOR + 7)

/**
 * @brief Property key to determine whether a backend supports scale-offset quantization encodings.
 *        See QNN_QUANTIZATION_ENCODING_SCALE_OFFSET.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_SCALE_OFFSET \
  (QNN_PROPERTY_GROUP_TENSOR + 8)

/**
 * @brief Property key to determine whether a backend supports axis scale-offset quantization
 *        encodings. See QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET \
  (QNN_PROPERTY_GROUP_TENSOR + 9)

/**
 * @brief Property key to determine whether a backend supports bit-width scale-offset quantization
 *        encodings. See QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BW_SCALE_OFFSET \
  (QNN_PROPERTY_GROUP_TENSOR + 10)

/**
 * @brief Property key to determine whether a backend supports bit-width axis scale-offset
 *        quantization encodings. See QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET \
  (QNN_PROPERTY_GROUP_TENSOR + 11)

/**
 * @brief Property key to determine whether a backend supports block quantization encodings. See
 *        QNN_QUANTIZATION_ENCODING_BLOCK.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCK (QNN_PROPERTY_GROUP_TENSOR + 12)

/**
 * @brief Property key to determine whether a backend supports blockwise expansion
 *        quantization encodings. See QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION \
  (QNN_PROPERTY_GROUP_TENSOR + 13)

/**
 * @brief Property key to determine whether a backend supports vector quantization encodings. See
 *        QNN_QUANTIZATION_ENCODING_VECTOR.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_VECTOR (QNN_PROPERTY_GROUP_TENSOR + 14)

/**
 * @brief Property key to determine whether a backend supports deferred loading of raw tensor data
 *        through a callback. See QNN_TENSORMEMTYPE_RETRIEVE_RAW.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_RETRIEVE_RAW (QNN_PROPERTY_GROUP_TENSOR + 15)

/**
 * @brief Property key for determining whether a backend supports optional application
 *        writable tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_OPTIONAL_APP_WRITE (QNN_PROPERTY_GROUP_TENSOR + 16)
/**
 * @brief Property key for determining whether a backend supports optional application
 *        readable tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_OPTIONAL_APP_READ (QNN_PROPERTY_GROUP_TENSOR + 17)
/**
 * @brief Property key for determining whether a backend supports optional application
 *        readable/writable tensors.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_OPTIONAL_APP_READWRITE (QNN_PROPERTY_GROUP_TENSOR + 18)

/**
 * @brief Property key for determining whether a backend supports MX data format
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_MX_DATA_FORMAT (QNN_PROPERTY_GROUP_TENSOR + 19)

/**
 *  @brief Property key for determining whether a backend supports
 *         QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888 data format.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UBWC_RGBA8888 (QNN_PROPERTY_GROUP_TENSOR + 20)

/**
 * @brief Property key for determining whether a backend supports QNN_TENSOR_DATA_FORMAT_UBWC_NV12
 *        data format.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UBWC_NV12 (QNN_PROPERTY_GROUP_TENSOR + 21)

/**
 * @brief Property key for determining whether a backend supports QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y
 *        data format.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UBWC_NV12_Y (QNN_PROPERTY_GROUP_TENSOR + 22)

/**
 * @brief Property key for determining whether a backend supports
 *        QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV data format.
 */
#define QNN_PROPERTY_TENSOR_SUPPORT_UBWC_NV12_UV (QNN_PROPERTY_GROUP_TENSOR + 23)

///
/// Definition of QNN_PROPERTY_GROUP_ERROR property group. This group is Optional portion of API.
///

/**
 * @brief Property key for the QNN Error API property group. This can be used as a key to
 *        check if Error API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_ERROR (QNN_PROPERTY_GROUP_CORE + 1000)

/**
 * @brief Property key to determine whether a backend supports retrieving verbose string descriptors
 *        of errorHandles. It determines support for QnnError_getVerboseMessage.
 */
#define QNN_PROPERTY_ERROR_GET_VERBOSE_MESSAGE (QNN_PROPERTY_GROUP_ERROR + 1)

///
/// Definition of QNN_PROPERTY_GROUP_MEMORY property group. This group is an optional API.
///

/**
 * @brief Property group for the QNN Memory API property group. This can be used as a key to
 *        check if Memory API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_MEMORY (QNN_PROPERTY_GROUP_CORE + 1100)

/**
 * @brief Property key to determine whether a backend supports ion memory type.
 */
#define QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_ION (QNN_PROPERTY_GROUP_MEMORY + 1)

/**
 * @brief Property key to determine whether a backend supports custom memory type.
 */
#define QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_CUSTOM (QNN_PROPERTY_GROUP_MEMORY + 2)

/**
 * @brief Property key to determine whether a backend supports DMA-BUF memory type.
 */
#define QNN_PROPERTY_MEMORY_SUPPORT_MEM_TYPE_DMA_BUF (QNN_PROPERTY_GROUP_MEMORY + 3)

///
/// Definition of QNN_PROPERTY_GROUP_SIGNAL property group. This group is an optional API.
///

/**
 * @brief Property group for signal support. This can be used as a key to
 *        check if Signal API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_SIGNAL (QNN_PROPERTY_GROUP_CORE + 1200)

/**
 * @brief Property key to determine whether a backend supports abort signals.
 */
#define QNN_PROPERTY_SIGNAL_SUPPORT_ABORT QNN_PROPERTY_GROUP_SIGNAL + 1

/**
 * @brief Property key to determine whether a backend supports timeout signals.
 */
#define QNN_PROPERTY_SIGNAL_SUPPORT_TIMEOUT QNN_PROPERTY_GROUP_SIGNAL + 2

///
/// Definition of QNN_PROPERTY_GROUP_LOG property group. This group is an optional API.
///

/**
 * @brief Property group for log support. This can be used as a key to
 *        check if Log API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_LOG (QNN_PROPERTY_GROUP_CORE + 1300)

/**
 * @brief Property key for determining whether a backend supports logging with the
 *        system's default stream (callback=NULL).
 */
#define QNN_PROPERTY_LOG_SUPPORTS_DEFAULT_STREAM (QNN_PROPERTY_GROUP_LOG + 1)

///
/// Definition of QNN_PROPERTY_GROUP_PROFILE property group. This group is an optional API.
///

/**
 * @brief Property group for profile support. This can be used as a key to
 *        check if Profile API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_PROFILE (QNN_PROPERTY_GROUP_CORE + 1400)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_PROFILE_CONFIG_OPTION_CUSTOM config option.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_CUSTOM_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 1)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS config option.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_MAX_EVENTS_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 2)

/**
 * @brief Property key for determining whether a backend supports querying extended event data. It
 *        determines support for QnnProfile_getExtendedEventData.
 */
#define QNN_PROPERTY_PROFILE_SUPPORTS_EXTENDED_EVENT (QNN_PROPERTY_GROUP_PROFILE + 3)

/**
 * @brief Property key for determining whether a backend supports the
 *        QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE config option.
 */
#define QNN_PROPERTY_PROFILE_SUPPORT_OPTRACE_CONFIG (QNN_PROPERTY_GROUP_PROFILE + 4)

/**
 * @brief Property group for device support. This can be used as a key to
 *        check if Device API is supported by a backend.
 */
#define QNN_PROPERTY_GROUP_DEVICE (QNN_PROPERTY_GROUP_CORE + 1500)

/**
 * @brief Property key for determining if a backend supports QnnDevice_getInfrastructure.
 */
#define QNN_PROPERTY_DEVICE_SUPPORT_INFRASTRUCTURE (QNN_PROPERTY_GROUP_DEVICE + 1)

///
/// Definition of QNN_PROPERTY_GROUP_CUSTOM property group. This group represents backend defined
/// properties.
///

/**
 * @brief Property group for custom backend properties.
 */
#define QNN_PROPERTY_GROUP_CUSTOM (QNN_PROPERTY_GROUP_CORE + 2000)

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief Type used for unique property identifiers.
 */
typedef uint32_t QnnProperty_Key_t;

/**
 * @brief QNN Property API result / error codes.
 */
typedef enum {
  QNN_PROPERTY_MIN_ERROR = QNN_MIN_ERROR_PROPERTY,
  //////////////////////////////////////////////

  QNN_PROPERTY_NO_ERROR = QNN_SUCCESS,
  /// Property in question is supported
  QNN_PROPERTY_SUPPORTED = QNN_SUCCESS,
  /// Property in question not supported.
  QNN_PROPERTY_NOT_SUPPORTED = QNN_COMMON_ERROR_NOT_SUPPORTED,

  // Remaining values signal errors.

  /// Backend did not recognize the property key.
  QNN_PROPERTY_ERROR_UNKNOWN_KEY = QNN_MIN_ERROR_PROPERTY + 0,

  //////////////////////////////////////////////
  QNN_PROPERTY_MAX_ERROR = QNN_MAX_ERROR_PROPERTY,
  // Unused, present to ensure 32 bits.
  QNN_PROPERTY_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnProperty_Error_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Queries a capability of the backend.
 *
 * @note Safe to call any time, backend does not have to be created.
 *
 * @param[in] key Key which identifies the capability within group.
 *
 * @return Error code:
 *         - QNN_PROPERTY_SUPPORTED: if the backend supports capability.
 *         - QNN_PROPERTY_ERROR_UNKNOWN_KEY: The provided key is not valid.
 *         - QNN_PROPERTY_NOT_SUPPORTED: if the backend does not support capability.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProperty_hasCapability(QnnProperty_Key_t key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
