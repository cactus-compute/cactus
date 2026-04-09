//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Context component API.
 *
 *          Requires Backend to be initialized.
 *          Graphs and Tensors are created within Context.
 *          Context content once created can be cached into a binary form.
 */

#ifndef QNN_CONTEXT_H
#define QNN_CONTEXT_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Context API result / error codes.
 */
typedef enum {
  QNN_CONTEXT_MIN_ERROR = QNN_MIN_ERROR_CONTEXT,
  ////////////////////////////////////////////

  /// Qnn context success
  QNN_CONTEXT_NO_ERROR = QNN_SUCCESS,
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Context-specific memory allocation/deallocation failure
  QNN_CONTEXT_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// An argument to QNN context API is deemed invalid by a backend
  QNN_CONTEXT_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_CONTEXT + 0,
  /// A QNN context has not yet been created in the backend
  QNN_CONTEXT_ERROR_CTX_DOES_NOT_EXIST = QNN_MIN_ERROR_CONTEXT + 1,
  /// Invalid/NULL QNN context handle
  QNN_CONTEXT_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_CONTEXT + 2,
  /// Attempting an operation when graphs in a context haven't been finalized
  QNN_CONTEXT_ERROR_NOT_FINALIZED = QNN_MIN_ERROR_CONTEXT + 3,
  /// Attempt to access context binary with an incompatible version
  QNN_CONTEXT_ERROR_BINARY_VERSION = QNN_MIN_ERROR_CONTEXT + 4,
  /// Failure to create context from binary
  QNN_CONTEXT_ERROR_CREATE_FROM_BINARY = QNN_MIN_ERROR_CONTEXT + 5,
  /// Failure to get size of a QNN serialized context
  QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED = QNN_MIN_ERROR_CONTEXT + 6,
  /// Failure to generate a QNN serialized context
  QNN_CONTEXT_ERROR_GET_BINARY_FAILED = QNN_MIN_ERROR_CONTEXT + 7,
  /// Invalid context binary configuration
  QNN_CONTEXT_ERROR_BINARY_CONFIGURATION = QNN_MIN_ERROR_CONTEXT + 8,
  /// Failure to set profile
  QNN_CONTEXT_ERROR_SET_PROFILE = QNN_MIN_ERROR_CONTEXT + 9,
  /// Invalid config
  QNN_CONTEXT_ERROR_INVALID_CONFIG = QNN_MIN_ERROR_CONTEXT + 10,
  /// Attempt to create a context from suboptimal binary
  QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL = QNN_MIN_ERROR_CONTEXT + 11,
  /// Call aborted early due to a QnnSignal_trigger call issued
  /// to the observed signal object.
  QNN_CONTEXT_ERROR_ABORTED = QNN_MIN_ERROR_CONTEXT + 12,
  /// Call aborted early due to a QnnSignal timeout
  QNN_CONTEXT_ERROR_TIMED_OUT = QNN_MIN_ERROR_CONTEXT + 13,
  /// Incremental Binary Buffer was not allocated by backend
  QNN_CONTEXT_ERROR_INCREMENT_INVALID_BUFFER = QNN_MIN_ERROR_CONTEXT + 14,
  ////////////////////////////////////////////
  QNN_CONTEXT_MAX_ERROR = QNN_MAX_ERROR_CONTEXT,
  // Unused, present to ensure 32 bits.
  QNN_CONTEXT_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnContext_Error_t;

/**
 * @brief Context specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnContext_CustomConfig_t;

/**
 * @brief This enum defines context config options.
 */
typedef enum {
  /// Sets context custom options via QnnContext_CustomConfig_t
  QNN_CONTEXT_CONFIG_OPTION_CUSTOM = 0,
  /// Sets the default priority for graphs in this context. QNN_GRAPH_CONFIG_OPTION_PRIORITY can be
  /// used to override this default.
  QNN_CONTEXT_CONFIG_OPTION_PRIORITY = 1,
  /// Sets the error reporting level.
  QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING = 2,
  /// Sets the string used for custom oem functionality. This config option is DEPRECATED.
  QNN_CONTEXT_CONFIG_OPTION_OEM_STRING = 3,
  /// Sets async execution queue depth for all graphs in this context. This option represents the
  /// number of executions that can be in the queue at a given time before QnnGraph_executeAsync()
  /// will start blocking until a new spot is available. Queue depth is subject to a maximum limit
  /// determined by the backend and available system resources. The default depth is
  /// backend-specific, refer to SDK documentation.
  QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH = 4,
  /// Null terminated array of null terminated strings listing the names of the graphs to
  /// deserialize from a context binary. All graphs are enabled by default. An error is generated if
  /// an invalid graph name is provided.
  QNN_CONTEXT_CONFIG_ENABLE_GRAPHS = 5,
  /// Sets the peak memory limit hint of a deserialized context in megabytes
  QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT = 6,
  /// Indicates that the context binary pointer is available during QnnContext_createFromBinary and
  /// until QnnContext_free is called.
  QNN_CONTEXT_CONFIG_PERSISTENT_BINARY = 7,
  /// Sets the context binary check type when reading binary caches
  QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY = 8,
  /// Defer graph deserialization till first graph retrieval
  QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT = 9,
  // Unused, present to ensure 32 bits.
  QNN_CONTEXT_CONFIG_UNDEFINED = 0x7FFFFFFF
} QnnContext_ConfigOption_t;

typedef enum {
  /// A binary cache is compatible if it could run on the device. This is the
  /// default.
  QNN_CONTEXT_BINARY_COMPATIBILITY_PERMISSIVE = 0,
  /// A binary cache is compatible if it could run on the device and fully
  /// utilize hardware capability, otherwise QnnContext_CreateFromBinary
  /// may return QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL.
  QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT = 1,
  // Unused, present to ensure 32 bits
  QNN_CONTEXT_BINARY_COMPATIBILITY_TYPE_UNDEFINED = 0x7FFFFFF
} QnnContext_BinaryCompatibilityType_t;

typedef enum {
  /// Sets a numeric value for the maximum queue depth
  QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_NUMERIC = 0,

  // Unused, present to ensure 32 bits
  QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED = 0x7FFFFFF
} QnnContext_AsyncExecutionQueueDepthType_t;

/**
 * @brief This struct provides async execution queue depth.
 */
typedef struct {
  QnnContext_AsyncExecutionQueueDepthType_t type;
  union UNNAMED {
    uint32_t depth;
  };
} QnnContext_AsyncExecutionQueueDepth_t;

/// QnnContext_AsyncExecutionQueueDepth_t initializer macro
#define QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_INIT                 \
  {                                                                  \
    QNN_CONTEXT_ASYNC_EXECUTION_QUEUE_DEPTH_TYPE_UNDEFINED, /*type*/ \
    {                                                                \
      0 /*depth*/                                                    \
    }                                                                \
  }

/**
 * @brief This struct provides context configuration.
 */
typedef struct {
  QnnContext_ConfigOption_t option;
  union UNNAMED {
    /// Used with QNN_CONTEXT_CONFIG_OPTION_CUSTOM.
    QnnContext_CustomConfig_t customConfig;
    /// Used with QNN_CONTEXT_CONFIG_OPTION_PRIORITY.
    Qnn_Priority_t priority;
    /// Used with QNN_CONTEXT_CONFIG_OPTION_ERROR_REPORTING.
    Qnn_ErrorReportingConfig_t errorConfig;
    /// DEPRECATED. Used with QNN_CONTEXT_CONFIG_OPTION_OEM_STRING
    const char* oemString;
    /// Used with QNN_CONTEXT_CONFIG_ASYNC_EXECUTION_QUEUE_DEPTH
    QnnContext_AsyncExecutionQueueDepth_t asyncExeQueueDepth;
    /// Used with QNN_CONTEXT_CONFIG_ENABLE_GRAPHS
    const char* const* enableGraphs;
    /// Used with QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT
    uint64_t memoryLimitHint;
    /// Used with QNN_CONTEXT_CONFIG_PERSISTENT_BINARY
    uint8_t isPersistentBinary;
    /// Used with QNN_CONTEXT_CONFIG_BINARY_COMPATIBILITY
    QnnContext_BinaryCompatibilityType_t binaryCompatibilityType;
    /// Used with QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT
    uint8_t isGraphInitDeferred;
  };
} QnnContext_Config_t;

/// QnnContext_Config_t initializer macro
#define QNN_CONTEXT_CONFIG_INIT              \
  {                                          \
    QNN_CONTEXT_CONFIG_UNDEFINED, /*option*/ \
    {                                        \
      NULL /*customConfig*/                  \
    }                                        \
  }

/**
 * @brief Enum to distinguish notify type
 */
typedef enum {
  // Graph initialization
  QNN_CONTEXT_NOTIFY_TYPE_GRAPH_INIT = 0,
  // Context initialization
  QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT = 1,
  /// Unused, present to ensure 32 bits.
  QNN_CONTEXT_NOTIFY_TYPE_UNDEFINED = 0x7FFFFFF
} QnnContext_createFromBinaryAsyncNotifyType_t;

/**
 * @brief A client-defined callback function.
 *
 * @param[in] context handle to a created context
 *
 * @param[in] graph handle to a created graph
 *
 * @param[in] graphName created graph's name
 *
 * @param[in] notifyType enum type indicating whether a context or a graph init is complete
 *
 * @param[in] notifyParam Client supplied data object which may be used to identify
 *                        which function this callback applies to.
 *
 * @param[in] status graph or context initialization result
 *
 * @return None
 *
 */
typedef void (*QnnContext_createFromBinaryNotifyFn_t)(
    Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t graph,
    const char* graphName,
    QnnContext_createFromBinaryAsyncNotifyType_t notifyType,
    void* notifyParam,
    Qnn_ErrorHandle_t status);

/**
 * @brief This structure serves as a consolidated representation of context-related parameters.
 *        QnnContext_createFromBinaryListAsync API takes a list of these parameters for initializing
 *        multiple context binaries.
 */
typedef struct {
  /// Config pointer to a NULL-terminated array of config option pointers for one context. NULL
  /// is allowed and indicates that no config options are provided. If not provided, all config
  /// options have default values consistent with the serialized context. If the same config option
  /// type is provided multiple times, the last option value will be used.
  const QnnContext_Config_t** config;
  /// A pointer to the context binary
  const void* binaryBuffer;
  /// Holds the size of the context binary
  const Qnn_ContextBinarySize_t binaryBufferSize;
  /// The profile handle on which metrics are populated and can be queried. Use a NULL handle
  /// to disable profile collection. If a handle is reused, it will reset and be populated with
  /// values from the current call.
  Qnn_ProfileHandle_t profile;
  /// Pointer to a notification function, cannot be NULL
  QnnContext_createFromBinaryNotifyFn_t notifyFunc;
  /// Client-supplied data object which will be passed back via _notifyFn_ and can be used to
  /// identify which context's asynchronous initialization instance the __notifyFn__ applies to.
  /// Can be NULL if client does not need it.
  void* notifyParam;
} QnnContext_ParamsV1_t;

/**
 * @brief Enum to distinguish various context params definitions
 */
typedef enum {
  QNN_CONTEXT_PARAMS_VERSION_1 = 1,
  /// Unused, present to ensure 32 bits.
  QNN_CONTEXT_PARAMS_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnContext_ParamsVersion_t;

/**
 * @brief Structure which provides various versions of context params
 */
typedef struct {
  QnnContext_ParamsVersion_t version;
  union UNNAMED {
    QnnContext_ParamsV1_t v1;
  };
} QnnContext_Params_t;

/**
 * @brief Enum to distinguish type of binary section to retrieve
 */
typedef enum {
  /// Portion of the context binary containing recent updates applied through
  /// QnnTensor_updateGraphTensors() or QnnTensor_updateContextTensors()
  QNN_CONTEXT_SECTION_UPDATABLE = 1,
  /// Portion of the context binary containing recent static data updates applied through
  /// QnnTensor_updateGraphTensors() or QnnTensor_updateContextTensors()
  QNN_CONTEXT_SECTION_UPDATABLE_WEIGHTS = 2,
  /// Portion of the context binary containing recent quantization updates applied through
  /// QnnTensor_updateGraphTensors() or QnnTensor_updateContextTensors()
  QNN_CONTEXT_SECTION_UPDATABLE_QUANT_PARAMS = 3,
  /// Unused, present to ensure 32 bits.
  QNN_CONTEXT_SECTION_UNDEFINED = 0x7FFFFFFF
} QnnContext_SectionType_t;

/**
 * @brief An enum specifying memory types of context binary data.
 */
typedef enum {
  /// Raw memory pointer
  QNN_CONTEXTMEMTYPE_RAW = 0,
  /// Memory object, provide capability for memory sharing in between QNN accelerator backends.
  QNN_CONTEXTMEMTYPE_MEMHANDLE = 1,
  // Unused, present to ensure 32 bits.
  QNN_CONTEXTMEMTYPE_UNDEFINED = 0x7FFFFFFF
} QnnContext_MemType_t;

/**
 * @brief A struct which defines a memory buffer
 */
typedef struct {
  /// app-accessible data pointer, provided by app.
  void* data;
  /// size of buffer, in bytes, pointed to by data.
  Qnn_ContextBinarySize_t dataSize;
} Qnn_BinaryBuffer_t;

/**
 * @brief This structure defines a client created binary buffer containing the context binary.
 *
 */
typedef struct {
  QnnContext_MemType_t memType;
  /// Actual data contained in the context binary.
  union UNNAMED {
    /// Context binary data provided by client as a pointer to raw memory (see
    /// QNN_CONTEXTMEMTYPE_RAW).
    Qnn_BinaryBuffer_t binaryBuf;
    /// Context binary data shared via a memory handle (see QNN_CONTEXTMEMTYPE_MEMHANDLE).
    Qnn_MemHandle_t memHandle;
  };
} QnnContext_BufferV1_t;

/**
 * @brief Enum to distinguish various context params definitions
 */
typedef enum {
  QNN_CONTEXT_BUFFER_VERSION_1 = 1,
  /// Unused, present to ensure 32 bits.
  QNN_CONTEXT_BUFFER_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnContext_BufferVersion_t;

/**
 * @brief Structure which provides various versions of context params
 */
typedef struct {
  QnnContext_BufferVersion_t version;
  union UNNAMED {
    QnnContext_BufferV1_t v1;
  };
} QnnContext_Buffer_t;

/**
 * @brief This enum defines context property options.
 */
typedef enum {
  /// Gets context custom properties, see backend specific documentation.
  QNN_CONTEXT_PROPERTY_OPTION_CUSTOM = 0,
  /// Value selected to ensure 32 bits.
  QNN_CONTEXT_PROPERTY_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnContext_PropertyOption_t;

/**
 * @brief Context specific object for custom property
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnContext_CustomProperty_t;

/**
 * @brief This struct provides context property.
 *        Option is specified by the client. Everything
 *        else is written by the backend.
 */
typedef struct {
  QnnContext_PropertyOption_t option;
  union UNNAMED {
    QnnContext_CustomProperty_t customProperty;
  };
} QnnContext_Property_t;

// clang-format off
/// QnnContext_Property_t initializer macro
#define QNN_CONTEXT_PROPERTY_INIT                     \
  {                                                   \
    QNN_CONTEXT_PROPERTY_OPTION_UNDEFINED, /*option*/ \
    {                                                 \
      NULL /*customProperty*/                         \
    }                                                 \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create a context.
 *        Context holds graphs, operations and tensors
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used.
 *
 * @param[out] context A handle to the created context.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: at least one argument is invalid
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: failure in allocating memory when creating context
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_ or _device_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: an optional feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_create(Qnn_BackendHandle_t backend,
                                    Qnn_DeviceHandle_t device,
                                    const QnnContext_Config_t** config,
                                    Qnn_ContextHandle_t* context);

/**
 * @brief A function to set/modify configuration options on an already generated context.
 *        Backends are not required to support this API.
 *
 * @param[in] context A context handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used. If a backend cannot support all
 *                   provided configs it will fail.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: at least one config option is invalid
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: an optional feature is not supported
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_setConfig(Qnn_ContextHandle_t context,
                                       const QnnContext_Config_t** config);

/**
 * @brief A function to get the size of memory to be allocated to hold
 *        the context content in binary (serialized) form.
 *        This function must be called after all entities in the context have been finalized.
 *
 * @param[in] context A context handle.
 *
 * @param[out] binaryBufferSize The amount of memory in bytes a client will need to allocate
 *                              to hold context content in binary form.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBufferSize_ is NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary size for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySize(Qnn_ContextHandle_t context,
                                           Qnn_ContextBinarySize_t* binaryBufferSize);

/**
 * @brief A function to get the context content in binary (serialized) form.
 *        The binary can be used to re-create context by using QnnContext_createFromBinary(). This
 *        function must be called after all entities in the context have been finalized. Unconsumed
 *        tensors are not included in the binary. Client is responsible for allocating sufficient
 *        and valid memory to hold serialized context content produced by this method. It is
 *        recommended the user calls QnnContext_getBinarySize() to allocate a buffer of sufficient
 *        space to hold the binary.
 *
 * @param[in] context A context handle.
 *
 * @param[in] binaryBuffer Pointer to the user-allocated context binary memory.
 *
 * @param[in] binaryBufferSize Size of _binaryBuffer_ to populate context binary with, in bytes.
 *
 * @param[out] writtenBufferSize Amount of memory actually written into _binaryBuffer_, in bytes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: one of the arguments to the API is invalid/NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getBinary(Qnn_ContextHandle_t context,
                                       void* binaryBuffer,
                                       Qnn_ContextBinarySize_t binaryBufferSize,
                                       Qnn_ContextBinarySize_t* writtenBufferSize);

/**
 * @brief A function to validate a stored binary.
 *        The binary was previously obtained via QnnContext_getBinary() and stored by a client.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. In case they are not provided,
 *                   all config options have a default value in accordance with the serialized
 *                   context. If the same config option type is provided multiple times, the last
 *                   option value will be used.
 *
 * @param[in] binaryBuffer A pointer to the context binary.
 *
 * @param[in] binaryBufferSize Holds the size of the context binary.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while validating binary cache
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to validate binary cache
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL: suboptimal binary is used when
 *           QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT is specified in the config option
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, or _device_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_validateBinary(Qnn_BackendHandle_t backend,
                                            Qnn_DeviceHandle_t device,
                                            const QnnContext_Config_t** config,
                                            const void* binaryBuffer,
                                            Qnn_ContextBinarySize_t binaryBufferSize);

/**
 * @brief A function to create a context from a stored binary.
 *        The binary was previously obtained via QnnContext_getBinary() and stored by a client. The
 *        content of a context created in this way cannot be further altered, meaning *no* new
 *        nodes or tensors can be added to the context. Creating context by deserializing provided
 *        binary is meant for fast content creation, ready to execute on.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. In case they are not provided,
 *                   all config options have a default value in accordance with the serialized
 *                   context. If the same config option type is provided multiple times, the last
 *                   option value will be used.
 *
 * @param[in] binaryBuffer A pointer to the context binary.
 *
 * @param[in] binaryBufferSize Holds the size of the context binary.
 *
 * @param[out] context A handle to the created context.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ or _context_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating context
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to deserialize binary and
 *           create context from it
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL: suboptimal binary is used when
 *           QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT is specified in the config option
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, __profile_, or _device_ is not a
 *           valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_createFromBinary(Qnn_BackendHandle_t backend,
                                              Qnn_DeviceHandle_t device,
                                              const QnnContext_Config_t** config,
                                              const void* binaryBuffer,
                                              Qnn_ContextBinarySize_t binaryBufferSize,
                                              Qnn_ContextHandle_t* context,
                                              Qnn_ProfileHandle_t profile);

/**
 * @brief A function to create a context from a stored binary, which supports control signals.
 *        The binary was previously obtained via QnnContext_getBinary() and stored by a client. The
 *        content of a context created in this way cannot be further altered, meaning *no* new
 *        nodes or tensors can be added to the context. Creating context by deserializing provided
 *        binary is meant for fast content creation, ready to execute on.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle to set hardware affinity for the created context. NULL value
 *                   can be supplied for device handle and it is equivalent to calling
 *                   QnnDevice_create() with NULL config.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. In case they are not provided,
 *                   all config options have a default value in accordance with the serialized
 *                   context. If the same config option type is provided multiple times, the last
 *                   option value will be used.
 *
 * @param[in] binaryBuffer A pointer to the context binary.
 *
 * @param[in] binaryBufferSize Holds the size of the context binary.
 *
 * @param[out] context A handle to the created context.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @param[in] signal Signal object to control the execution of the create context from binary
 *                   process. NULL may be passed to indicate that no execution control is requested,
 *                   and the create operation should continue to completion uninterrupted.
 *                   The signal object, if not NULL, is considered to be in-use for
 *                   the duration of the call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ or _context_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating context
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to deserialize binary and
 *           create context from it
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL: suboptimal binary is used when
 *           QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT is specified in the config option
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, __profile_, or _device_ is not a
 *           valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *         - QNN_CONTEXT_ERROR_ABORTED: the call is aborted before completion due to user
 *           cancellation
 *         - QNN_CONTEXT_ERROR_TIMED_OUT: the call is aborted before completion due to a timeout
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_createFromBinaryWithSignal(Qnn_BackendHandle_t backend,
                                                        Qnn_DeviceHandle_t device,
                                                        const QnnContext_Config_t** config,
                                                        const void* binaryBuffer,
                                                        Qnn_ContextBinarySize_t binaryBufferSize,
                                                        Qnn_ContextHandle_t* context,
                                                        Qnn_ProfileHandle_t profile,
                                                        Qnn_SignalHandle_t signal);

/**
 * @brief The purpose of this function is to asynchronously create multiple contexts from binaries
 *        in a single API call. The API can be used with QnnSignal. When the function is invoked,
 *        the deserialization/initialization of each context will occur in the background. As soon
 *        as a graph reaches an executable state (i.e., initialization is completed), the client
 *        will receive a notification via the specified notification function. Once notification is
 *        received, the client can proceed to execute the graph. If a context contains multiple
 *        graphs, there will be multiple callbacks through the notification function.
 *
 * @note: Until a notification is received indicating that at least one graph or context is in an
 *        executable state, other context or graph-based functions cannot be called.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] device A device handle used to set hardware affinity for the created context. A
 *                   NULL value can be supplied for the device handle, which is equivalent to
 *                   calling QnnDevice_create() with a NULL config.
 *
 * @param[in] contextParams Pointer to a NULL-terminated array of context parameters.
 *
 * @param[in] listConfig Config pointer to a NULL-terminated array of config option pointers that
 *                       apply to all contexts in the list. NULL is allowed and indicates that no
 *                       config options are provided. If not provided, all config options have
 *                       default values consistent with the serialized context. If the same config
 *                       option type is provided multiple times, the last option value will be used.
 *                       listConfig will override options also specified in contextParams.
 *
 * @param[in] signal Signal object to control the create context from binary process.
 *                   NULL may be passed to indicate that no execution control is requested,
 *                   and the create operation should continue to completion uninterrupted.
 *                   The signal object, if not NULL, is considered to be in-use for
 *                   the duration of the call including the asynchronous
 *                   deserialization/initialization.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: contextParams is empty or any individual
 *           contextParam's _binaryBuffer_ or notifyFunc is NULL or binaryBufferSize is 0
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating any context
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to deserialize any binary and create
 *           context from it
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL: suboptimal binary is used when
 *           QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT is specified in the config option
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, _device_, __signal__ or any individual
 *           contextParam's _profile_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values in either listConfig or
 *           any individual contextParam's config is invalid
 *         - QNN_CONTEXT_ERROR_ABORTED: the call is aborted before completion due to user
 *           cancellation including during any individual asynchronous
 *           deserialization/initialization
 *         - QNN_CONTEXT_ERROR_TIMED_OUT: the call is aborted before completion due to a timeout
 *           including during any individual asynchronous deserialization/initialization
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_createFromBinaryListAsync(Qnn_BackendHandle_t backend,
                                                       Qnn_DeviceHandle_t device,
                                                       const QnnContext_Params_t** contextParams,
                                                       const QnnContext_Config_t** listConfig,
                                                       Qnn_SignalHandle_t signal);

/**
 * @brief A function to finish context creation when originally created with deffered
 *        graph initialization enabled (see QNN_CONTEXT_CONFIG_OPTION_DEFER_GRAPH_INIT)
 *
 * @param[in] context A context handle.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ or _context_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating context
 *         - QNN_CONTEXT_ERROR_CREATE_FROM_BINARY: failed to deserialize binary and
 *           create context from it
 *         - QNN_CONTEXT_ERROR_BINARY_VERSION: incompatible version of the binary
 *         - QNN_CONTEXT_ERROR_BINARY_CONFIGURATION: binary is not configured for this device
 *         - QNN_CONTEXT_ERROR_BINARY_SUBOPTIMAL: suboptimal binary is used when
 *           QNN_CONTEXT_BINARY_COMPATIBILITY_STRICT is specified in the config option
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, __profile_, or _device_ is not a
 *           valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_CONFIG: one or more config values is invalid
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_finalize(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile);

/**
 * @brief Retrieve a section of the binary as specified by __section__. The size of this section
 *        depends on the type of section requested. For example, for QNN_CONTEXT_SECTION_UPDATABLE
 *        sections, this will have all the updatable tensor information.
 *
 * @param[in] context A context handle.
 *
 * @param[in] graph A graph handle. This argument is optional. When supplied the return size only
 *                  applies to the size of the context binary section pertaining to this graph. When
 *                  excluded the returned binary contains associated updates to all graphs in the
 *                  context. Some backends may require _graph_ as an argument. Support is determined
 *                  by QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_SECTION_FULL_CONTEXT.
 *
 * @param[in] section The section of the binary to retrieve.
 *
 * @param[out] binaryBufferSize The amount of memory in bytes a client will need to allocate
 *                              to hold context content updates in binary form.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBufferSize_ is NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_SIZE_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary size for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */

QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySectionSize(Qnn_ContextHandle_t context,
                                                  Qnn_GraphHandle_t graph,
                                                  QnnContext_SectionType_t section,
                                                  Qnn_ContextBinarySize_t* binaryBufferSize);

/**
 * @brief Retrieve section of the context binary. Content of the section is specified by
 *        __section__. The size of the section is retrieved from QnnContext_getBinarySectionSize().
 *
 * @param[in] context A context handle.
 *
 * @param[in] binaryBuffer Pointer to the user-allocated context binary memory.
 *
 * @param[in] graph A graph handle. This argument is optional. When supplied the returned binary
 *                  only contains the context binary section pertaining to this graph. When excluded
 *                  the returned binary contains associated updates to all graphs in the context.
 *                  Some backends may require _graph_ as an argument. Support is determined by
 *                  QNN_PROPERTY_CONTEXT_SUPPORT_BINARY_SECTION_FULL_CONTEXT.
 *
 * @param[in] section The section of the binary to retrieve. When section is
 *                    QNN_CONTEXT_SECTION_UPDATABLE the returned binary will contain all of the
 *                    updatable tensors associated with the context and graph combination. Binary
 *                    sections of type QNN_CONTEXT_SECTION_UPDATABLE have Qnn System Context
 *                    metadata containing information about any modified input and output tensors,
 *                    and therefore may be used with QnnSystemContext_getMetadata() and
 *                    QnnSystemContext_getBinaryInfo().
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @param[in] signal Signal object to control the execution of the create context from binary
 *                   process. NULL may be passed to indicate that no execution control is requested,
 *                   and the create operation should continue to completion uninterrupted.
 *                   The signal object, if not NULL, is considered to be in-use for
 *                   the duration of the call.
 *
 * @param[out] writtenBufferSize Amount of memory actually written into _binaryBuffer_, in bytes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: one of the arguments to the API is invalid/NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary update for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getBinarySection(Qnn_ContextHandle_t context,
                                              Qnn_GraphHandle_t graph,
                                              QnnContext_SectionType_t section,
                                              const QnnContext_Buffer_t* binaryBuffer,
                                              Qnn_ContextBinarySize_t* writtenBufferSize,
                                              Qnn_ProfileHandle_t profile,
                                              Qnn_SignalHandle_t signal);

/**
 * @brief Apply a section to the contextBinary produced by a prior QnnContext_getBinarySection()
 *        call. If successful, this section overwrites previously applied sections. If the call to
 *        applyBinarySection() fails, it indicates the changes were not applied, and that the
 *        context retains its prior state. In this case the context is still valid and may be used
 *        for subsequent inferences.
 *
 * @param[in] context A context handle.
 *
 * @param[in] graph A graph handle. This argument is optional. When supplied the returned binary
 *                  only contains the context binary section pertaining to this graph. When excluded
 *                  the returned binary contains associated updates to all graphs in the context.
 *
 * @param[in] section The section of the binary to retrieve. When section is
 *                    QNN_CONTEXT_SECTION_UPDATABLE the returned binary will contain all of the
 *                    updatable tensors associated with the context and graph combination.
 *
 * @param[in] binaryBuffer Pointer to the user-allocated context binary memory.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @param[in] signal Signal object to control the execution of the create context from binary
 *                   process. NULL may be passed to indicate that no execution control is requested,
 *                   and the create operation should continue to completion uninterrupted.
 *                   The signal object, if not NULL, is considered to be in-use for
 *                   the duration of the call.
 *
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _binaryBuffer_ or _context_ is NULL
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: memory allocation error while creating context update
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _backend_, __profile_, or _signal_ is not a
 *           valid handle
 *
 * @note Use corresponding API through QnnInterface_t.
 * @note When using this API with QNN_CONTEXT_CONFIG_PERSISTENT_BINARY enabled,
 *       binaryBuffer should be available and persistent from first call to
 *       QnnContext_applyBinarySection until QnnContext_free.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_applyBinarySection(Qnn_ContextHandle_t context,
                                                Qnn_GraphHandle_t graph,
                                                QnnContext_SectionType_t section,
                                                const QnnContext_Buffer_t* binaryBuffer,
                                                Qnn_ProfileHandle_t profile,
                                                Qnn_SignalHandle_t signal);

/**
 * @brief A function to get a list of context properties.
 *        Backends are not required to support this API.
 *
 * @param[in] contextHandle A context handle.
 *
 * @param[in/out] properties Pointer to a null terminated array of pointers containing the
 *                           properties associated with the passed contextHandle. Memory for
 *                           this information is owned and managed by the client. Client
 *                           needs to populate the property options being requested. If
 *                           _contextHandle_ is not recognized, the pointer _properties_
 *                           points to is set to nullptr.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE: _contextHandle_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _properties_ is NULL or at least one property
 *           option is invalid
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: at least one valid property option is not
 *           supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getProperty(Qnn_ContextHandle_t contextHandle,
                                         QnnContext_Property_t** properties);

/**
 * @brief A function to get the next piece of the context binary, incrementally produced from the
 *        backend. The backend returns a pointer to constant data which it owns, the data's size and
 *        the starting offset where the incremental binary buffer begins. The memory provided here
 *        must be released through QnnContext_releaseIncrementalBinary. Incremental pieces of the
 *        context binary may be provided in random order i.e. startOffset is independent of previous
 *        calls.
 *
 *        @note modifications made to the context in between calls to
 *        QnnContext_getIncrementalBinary results in undefined behavior.
 *
 * @param[in] context A context handle.
 *
 * @param[out] binaryBuffer Pointer to backend provided/owned buffer
 *
 * @param[out] startOffset Starting offset for binary data.
 *
 * @param[out] writtenBufferSize Amount of memory actually written into _binaryBuffer_, in bytes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: one of the arguments to the API is invalid/NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_CONTEXT_ERROR_GET_BINARY_FAILED: Operation failure due to other factors
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_getIncrementalBinary(Qnn_ContextHandle_t context,
                                                  const void** binaryBuffer,
                                                  Qnn_ContextBinarySize_t* startOffset,
                                                  Qnn_ContextBinarySize_t* writtenBufferSize);
/**
 * @brief A function to release a incrementally allocated portion of the context binary
 *        retrieved from a previous call to QnnContext_getIncrementalBinary.
 *
 * @param[in] context A context handle.
 *
 * @param[out] binaryBuffer Pointer to backend provided/owned buffer
 *
 * @param[out] startOffset Starting offset for binary data.
 *
 * @param[out] writtenBufferSize Amount of memory actually written into _binaryBuffer_, in bytes.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_UNSUPPORTED_FEATURE: a feature is not supported
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: one of the arguments to the API is invalid/NULL
 *         - QNN_CONTEXT_ERROR_NOT_FINALIZED: if there were any non-finalized entities in the
 *           context
 *         - QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED: Attempting to get binary for a
 *           context re-created from a cached binary.
 *         - QNN_CONTEXT_ERROR_INCREMENT_INVALID_BUFFER: The buffer __binaryBuffer__ starting at
 *           __startOffset__ was not allocated by the backend.
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: Not enough memory is available to retrieve the context
 *           content.
 *
 * @note Use corresponding API through QnnInterface_t.
 * */
QNN_API
Qnn_ErrorHandle_t QnnContext_releaseIncrementalBinary(Qnn_ContextHandle_t context,
                                                      const void* binaryBuffer,
                                                      Qnn_ContextBinarySize_t startOffset);

/**
 * @brief A function to free the context and all associated graphs, operations & tensors
 *
 * @param[in] context A context handle.
 *
 * @param[in] profile The profile handle on which metrics are populated and can be queried. Use
 *                    NULL handle to disable profile collection. A handle being re-used would reset
 *                    and is populated with values from the current call.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_CONTEXT_ERROR_INVALID_ARGUMENT: _profile_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_INVALID_HANDLE:  _context_ is not a valid handle
 *         - QNN_CONTEXT_ERROR_MEM_ALLOC: an error is encountered with de-allocation of associated
 *           memory
 *         - QNN_CONTEXT_ERROR_SET_PROFILE: failed to set profiling info
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnContext_free(Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_CONTEXT_H
