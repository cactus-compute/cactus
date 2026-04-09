//=============================================================================
//
//  Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Backend component API.
 *
 *          This is top level QNN API component.
 *          Most of the QNN API requires backend to be created first.
 */

#ifndef QNN_BACKEND_H
#define QNN_BACKEND_H

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
 * @brief QNN Backend API result / error codes.
 */
typedef enum {
  QNN_BACKEND_MIN_ERROR = QNN_MIN_ERROR_BACKEND,
  ////////////////////////////////////////////

  /// Qnn Backend success
  QNN_BACKEND_NO_ERROR = QNN_SUCCESS,
  /// General error relating to memory allocation in Backend API
  QNN_BACKEND_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// Backend attempted to be created on an unsupported platform
  QNN_BACKEND_ERROR_UNSUPPORTED_PLATFORM = QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED,
  /// Backend failed to initialize
  QNN_BACKEND_ERROR_CANNOT_INITIALIZE = QNN_MIN_ERROR_BACKEND + 0,
  /// Failed to free allocated resources during termination
  QNN_BACKEND_ERROR_TERMINATE_FAILED = QNN_MIN_ERROR_BACKEND + 2,
  /// Backend does not support requested functionality
  QNN_BACKEND_ERROR_NOT_SUPPORTED = QNN_MIN_ERROR_BACKEND + 3,
  /// Invalid function argument
  QNN_BACKEND_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_BACKEND + 4,
  /// Could not find specified op package
  QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND = QNN_MIN_ERROR_BACKEND + 5,
  /// Could not load interface provider from op package library
  QNN_BACKEND_ERROR_OP_PACKAGE_IF_PROVIDER_NOT_FOUND = QNN_MIN_ERROR_BACKEND + 6,
  /// Failed to register op package
  QNN_BACKEND_ERROR_OP_PACKAGE_REGISTRATION_FAILED = QNN_MIN_ERROR_BACKEND + 7,
  /// Backend does not support the op config's interface version
  QNN_BACKEND_ERROR_OP_PACKAGE_UNSUPPORTED_VERSION = QNN_MIN_ERROR_BACKEND + 8,
  /// An Op with the same package name and op name was already registered
  QNN_BACKEND_ERROR_OP_PACKAGE_DUPLICATE = QNN_MIN_ERROR_BACKEND + 9,
  /// Inconsistent backend configuration
  QNN_BACKEND_ERROR_INCONSISTENT_CONFIG = QNN_MIN_ERROR_BACKEND + 10,
  /// Invalid backend handle
  QNN_BACKEND_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_BACKEND + 11,
  /// Invalid config
  QNN_BACKEND_ERROR_INVALID_CONFIG = QNN_MIN_ERROR_BACKEND + 12,
  ////////////////////////////////////////////
  QNN_BACKEND_MAX_ERROR = QNN_MAX_ERROR_BACKEND,
  // Unused, present to ensure 32 bits.
  QNN_BACKEND_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnBackend_Error_t;

/**
 * @brief Backend specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnBackend_CustomConfig_t;

/**
 * @brief This enum defines backend config options.
 */
typedef enum {
  /// Sets backend custom options via QnnBackend_CustomConfig_t
  QNN_BACKEND_CONFIG_OPTION_CUSTOM = 0,
  /// Sets error reporting level
  QNN_BACKEND_CONFIG_OPTION_ERROR_REPORTING = 1,
  /// Key-value pair of platform options.
  QNN_BACKEND_CONFIG_OPTION_PLATFORM = 2,
  // Unused, present to ensure 32 bits.
  QNN_BACKEND_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnBackend_ConfigOption_t;

/**
 * @brief This struct provides backend configuration.
 */
typedef struct {
  QnnBackend_ConfigOption_t option;
  union UNNAMED {
    QnnBackend_CustomConfig_t customConfig;
    /// Applies error reporting configuration across backend.
    /// All QNN contexts share this common error configuration
    /// for APIs that are independent of a context.
    Qnn_ErrorReportingConfig_t errorConfig;
    /// Null-terminated platform option key-value pair. Multiple platform options can be specified.
    /// Max length is 1024.
    const char* platformOption;
  };
} QnnBackend_Config_t;

/// QnnBackend_Config_t initializer macro
#define QNN_BACKEND_CONFIG_INIT                     \
  {                                                 \
    QNN_BACKEND_CONFIG_OPTION_UNDEFINED, /*option*/ \
    {                                               \
      NULL /*customConfig*/                         \
    }                                               \
  }

/**
 * @brief Struct which encapsulates the fully-qualified name of an operation.
 */
typedef struct {
  /// The op package to which the operation belongs.
  const char* packageName;
  /// The type name of the operation.
  const char* name;
  /// The intended target platform for the combination of domain and operation name.
  /// Target may be unused (NULL) by some backends.
  const char* target;
} QnnBackend_OperationName_t;

// clang-format off
/// QnnBackend_OperationName_t initializer macro
#define QNN_BACKEND_OPERATION_NAME_INIT \
  {                                     \
    NULL,     /*packageName*/           \
    NULL,     /*name*/                  \
    NULL      /*target*/                \
  }
// clang-format on

/**
 * @brief This enum defines backend property options.
 */
typedef enum {
  /// Gets backend custom properties, see backend specific documentation.
  QNN_BACKEND_PROPERTY_OPTION_CUSTOM = 0,
  /// Value selected to ensure 32 bits.
  QNN_BACKEND_PROPERTY_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnBackend_PropertyOption_t;

/**
 * @brief Backend specific object for custom property
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnBackend_CustomProperty_t;

/**
 * @brief This struct provides backend property.
 *        Option is specified by the client. Everything
 *        else is written by the backend.
 */
typedef struct {
  QnnBackend_PropertyOption_t option;
  union UNNAMED {
    QnnBackend_CustomProperty_t customProperty;
  };
} QnnBackend_Property_t;

// clang-format off
/// QnnBackend_Property_t initializer macro
#define QNN_BACKEND_PROPERTY_INIT                     \
  {                                                   \
    QNN_BACKEND_PROPERTY_OPTION_UNDEFINED, /*option*/ \
    {                                                 \
      NULL /*customProperty*/                         \
    }                                                 \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Initialize a backend library and create a backend handle. Function is re-entrant and
 *        thread-safe.
 *
 * @param[in] logger A handle to the logger, use NULL handle to disable logging.
 *                   QnnBackend doesn't manage the lifecycle of logger and must be freed by using
 *                   QnnLog_free().
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers.
 *                   NULL is allowed and indicates no config options are provided.
 *                   All config options have default value, in case not provided.
 *                   If same config option type is provided multiple times,
 *                   the last option value will be used.
 *
 * @param[out] backend A handle to the created backend.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_BACKEND_ERROR_UNSUPPORTED_PLATFORM: Backend attempted to be created on
 *           unsupported platform
 *         - QNN_BACKEND_ERROR_INCONSISTENT_CONFIG: One or more backend configurations are
 *           inconsistent between multiple create calls. Refer to backend headers for which
 *           configuration options must be consistent.
 *         - QNN_BACKEND_ERROR_CANNOT_INITIALIZE: backend failed to initialize
 *         - QNN_BACKEND_ERROR_MEM_ALLOC: error related to memory allocation
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _logger_ is not a valid handle
 *         - QNN_BACKEND_ERROR_INVALID_CONFIG: one or more config values is invalid
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: an optional feature is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_create(Qnn_LogHandle_t logger,
                                    const QnnBackend_Config_t** config,
                                    Qnn_BackendHandle_t* backend);
/**
 * @brief A function to set/modify configuration options on an already generated backend.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers.
 *                   NULL is allowed and indicates no config options are provided.
 *                   All config options have default value, in case not provided.
 *                   If same config option type is provided multiple times,
 *                   the last option value will be used.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *         - QNN_BACKEND_ERROR_INVALID_CONFIG: at least one config option is invalid
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: an optional feature is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_setConfig(Qnn_BackendHandle_t backend,
                                       const QnnBackend_Config_t** config);

/**
 * @brief Get the QNN API version.
 *
 * @note Safe to call any time, backend does not have to be created.
 *
 * @param[out] pVersion Pointer to version object.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_BACKEND_ERROR_INVALID_ARGUMENT: if _pVersion_ was NULL
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_getApiVersion(Qnn_ApiVersion_t* pVersion);

/**
 * @brief Get build id for backend library.
 *
 * @note Safe to call any time, backend does not have to be created.
 *
 * @param[out] id Pointer to string containing the build id.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: No build ID is available
 *         - QNN_BACKEND_ERROR_INVALID_ARGUMENT: if _id_ is NULL
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_getBuildId(const char** id);

/**
 * @brief Register an operation package with the backend handle.
 *
 * @param[in] backend  A backend handle.
 *
 * @param[in] packagePath Path on disk to the op package library to load.
 *
 * @param[in] interfaceProvider The name of a function in the op package library which satisfies
 *                              the QnnOpPackage_InterfaceProvider_t interface. The backend will
 *                              use this function to retrieve the op package's interface.
 *
 * @param[in] target An optional parameter specifying the target platform on which the backend must
 *                   register the op package. Required in scenarios where an op package is to be
 *                   loaded on a processing unit that is different from the target on which the
 *                   backend runs. Ex: loading a DSP op package on ARM for optional online context
 *                   caching. Refer to additional documentation for a list of permissible target
 *                   names.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_BACKEND_ERROR_INVALID_ARGUMENT: if _packagePath_ or _interfaceProvider_ is NULL
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND: Could not open _packagePath_
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_IF_PROVIDER_NOT_FOUND: Could not find _interfaceProvider_
 *           symbol in package library
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_REGISTRATION_FAILED: Op package registration failed
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_UNSUPPORTED_VERSION: Op package has interface version not
 *           supported by this backend
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: Op package registration is not supported.
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_DUPLICATE: OpPackageName+OpName must be unique.
 *           Op package content information can be be obtained with QnnOpPackage interface.
 *           Indicates that an Op with the same package name and op name was already registered.
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_registerOpPackage(Qnn_BackendHandle_t backend,
                                               const char* packagePath,
                                               const char* interfaceProvider,
                                               const char* target);

/**
 * @brief Get the supported operations registered to a backend handle including built-in ops.
 *
 * @param[in] backend A backend handle. Can be NULL to obtain the built-in op package.
 *
 * @param[out] numOperations Number of supported operations.
 *
 * @param[out] operations Array of operation names. Memory is backend owned and de-allocated
 *                        during QnnBackend_free.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_BACKEND_ERROR_INVALID_ARGUMENT: if _numOperations_ or _operations_ is NULL
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_getSupportedOperations(Qnn_BackendHandle_t backend,
                                                    uint32_t* numOperations,
                                                    const QnnBackend_OperationName_t** operations);

/**
 * @brief A method to validate op config with an appropriate op package
 *        This is a wrapper API around the actual OpPackage interface method
 *        that performs op validation. Backend may pick an appropriate op package
 *        among ones that are registered with it for validation based on the attributes
 *        of the op configuration.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] opConfig Fully qualified struct containing the configuration of the operation.
 *
 * @note  _inputTensors_ and _outputTensors_ inside opConfig must be fully qualified for
 *        complete validation. However, their IDs (_id_) and names (_name_) are ignored during
 *        validation.
 *
 * @return Error code
 *         - QNN_SUCCESS if validation is successful
 *         - QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE: op config validation failed
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: Validation API not supported
 *         - QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND: No op package with matching
 *           op config attributes found.
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_validateOpConfig(Qnn_BackendHandle_t backend, Qnn_OpConfig_t opConfig);

/**
 * @brief A function to get a list of backend properties.
 *        Backends are not required to support this API.
 *
 * @param[in] backendHandle A backend handle.
 *
 * @param[in/out] properties Pointer to a null terminated array of pointers containing the
 *                           properties associated with the passed backendHandle. Memory for
 *                           this information is owned and managed by the client. Client
 *                           needs to populate the property options being requested. If
 *                           _contextHandle_ is not recognized, the pointer _properties_
 *                           points to is set to nullptr.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backendHandle_ is not a valid handle
 *         - QNN_BACKEND_ERROR_INVALID_ARGUMENT: _properties_ is NULL or at least one property option
 *           is invalid
 *         - QNN_BACKEND_ERROR_NOT_SUPPORTED: at least one valid property option is not
 *           supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_getProperty(Qnn_BackendHandle_t backendHandle,
                                         QnnBackend_Property_t** properties);

/**
 * @brief Free all resources associated with a backend handle.
 *
 * @param[in] backend handle to be freed.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered.
 *         - QNN_BACKEND_ERROR_MEM_ALLOC: error related to memory deallocation
 *         - QNN_BACKEND_ERROR_TERMINATE_FAILED: indicates failure to free
 *           resources or failure to invalidate handles and pointers allocated
 *           by the library
 *         - QNN_BACKEND_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnBackend_free(Qnn_BackendHandle_t backend);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
