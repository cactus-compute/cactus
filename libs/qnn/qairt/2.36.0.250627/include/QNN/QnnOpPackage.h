//=============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  QNN Operation Package API
 *
 *          Provides interface to the backend to use registered OpPackage libraries.
 */

#ifndef QNN_OP_PACKAGE_H
#define QNN_OP_PACKAGE_H

#include "QnnCommon.h"
#include "QnnLog.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

#define QNN_OP_PACKAGE_RESERVED_INFO_SIZE 12

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief A typedef for op package handles.
 */
typedef Qnn_Handle_t Qnn_OpPackageHandle_t;

/**
 * @brief Backend-defined and -provided infrastructure object which provides the package
 *        access to backend-wide facilities, e.g. memory management.
 */
typedef struct _QnnOpPackage_GlobalInfrastructure_t* QnnOpPackage_GlobalInfrastructure_t;

/**
 * @brief Backend-defined and -provided infrastructure object which provides the package
 *        access to graph-specific facilities, e.g. execution context or graph structure
 *        manipulation methods.
 */
typedef struct _QnnOpPackage_GraphInfrastructure_t* QnnOpPackage_GraphInfrastructure_t;

/**
 * @brief Backend-defined structure which represents Op implementation, with content
 *        executable within the context of a backend. Provided and managed by the package.
 */
typedef struct _QnnOpPackage_OpImpl_t* QnnOpPackage_OpImpl_t;

/**
 * @brief Backend-defined structure which contains the parameters and connectivity information
 *        for an operation node.
 */
typedef struct _QnnOpPackage_Node_t* QnnOpPackage_Node_t;

/**
 * @brief Backend-defined structure which encapsulates a graph optimization. Provided by the
 *        package to the backend to enable the backend to optimize graphs containing
 *        operation nodes for operations defined by this package.
 */
typedef struct _QnnOpPackage_Optimization_t* QnnOpPackage_Optimization_t;

/**
 * @brief Backend-defined structure which contains information for an operation.
 *        Provided by the package to the backend to convey information needed to properly
 *        construct an operation.
 */
typedef struct _QnnOpPackage_OperationInfo_t* QnnOpPackage_OperationInfo_t;

/**
 * @brief Backend-defined structure which contains information about Op package.
 *        Provided by the package to the backend to convey information needed to properly
 *        use the package.
 */
typedef struct _QnnOpPackage_PackageInfo_t QnnOpPackage_PackageInfo_t;

/**
 * @brief QNN OpPackage API result / error codes.
 */
typedef enum {
  QNN_OP_PACKAGE_MIN_ERROR = QNN_MIN_ERROR_OP_PACKAGE,
  //////////////////////////////////////////////////

  QNN_OP_PACKAGE_NO_ERROR = QNN_SUCCESS,
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Op package library was already initialized.
  QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED = QNN_MIN_ERROR_OP_PACKAGE + 0,
  /// Attempt to call a function in an uninitialized op package library.
  QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED = QNN_MIN_ERROR_OP_PACKAGE + 1,
  /// An invalid op package handle was provided.
  QNN_OP_PACKAGE_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_OP_PACKAGE + 2,
  /// Invalid infrastructure object used in initializing op package
  QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE = QNN_MIN_ERROR_OP_PACKAGE + 100,
  /// Invalid op package info object used in initializing op package
  QNN_OP_PACKAGE_ERROR_INVALID_INFO = QNN_MIN_ERROR_OP_PACKAGE + 101,
  /// Op configuration failed validation
  QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE = QNN_MIN_ERROR_OP_PACKAGE + 110,
  /// Invalid function argument
  QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_OP_PACKAGE + 200,
  /// Indicates an error has occurred due to a condition unforeseen by QNN, and possibly
  /// meaningful only in the context of the particular op package. Unless otherwise
  /// noted, any op package function may return this error.
  QNN_OP_PACKAGE_ERROR_GENERAL = QNN_COMMON_ERROR_GENERAL,

  //////////////////////////////////////////////////
  QNN_OP_PACKAGE_MAX_ERROR = QNN_MAX_ERROR_OP_PACKAGE,
  // Unused, present to ensure 32 bits.
  QNN_OP_PACKAGE_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnOpPackage_Error_t;

/**
 * @brief Struct describing the contents of an Op package.
 *        \n Reported to the backend by QnnOpPackage_GetInfoFn_t.
 */
typedef struct {
  /// Op package name. Must not be NULL nor empty string.
  const char* packageName;
  /// Array holding names of operations provided by the op package. Must not be NULL.
  /// Number of elements in the array is specified with _numOperations_.
  const char** operationNames;
  /// Array holding backend-defined operation information.
  /// This is optional, backend-specific information. Can be NULL.
  /// If not NULL, number of elements in the array is specified with _numOperations_.
  const QnnOpPackage_OperationInfo_t* operationInfo;
  /// Number of elements in _operationNames_ and _operationInfo_ arrays.
  uint32_t numOperations;
  /// Array holding backend-defined graph optimizations.
  /// This is optional, backend-specific information. Can be NULL.
  /// If not NULL, number of elements in the array is specified with _numOptimizations_.
  const QnnOpPackage_Optimization_t* optimizations;
  /// Number of elements in _optimizations_ array.
  uint32_t numOptimizations;
  /// BuildId (as returned by QnnBackend_getBuildId(), also see QNN_SDK_BUILD_ID)
  /// from QNN SDK which was used to create this OpPackage with. Allowed to be NULL.
  const char* sdkBuildId;
  /// API Version (as returned by QnnBackend_getApiVersion()) from QNN SDK which was
  /// used to create this OpPackage with. Allowed to be NULL.
  const Qnn_ApiVersion_t* sdkApiVersion;
  /// Op package level information. Allowed to be NULL.
  const QnnOpPackage_PackageInfo_t* packageInfo;
  /// Version of the set of operations implemented in the op package.
  const Qnn_Version_t* opsetVersion;
  /// Reserved for future extensibility. Must be memset to 0.
  size_t reserved[QNN_OP_PACKAGE_RESERVED_INFO_SIZE];
} QnnOpPackage_Info_t;

// clang-format off
/// QnnOpPackage_Info_t initializer macro
#define QNN_OP_PACKAGE_INFO_INIT   \
  {                                \
    NULL,     /*packageName*/      \
    NULL,     /*operationNames*/   \
    NULL,     /*operationInfo*/    \
    0u,       /*numOperations*/    \
    NULL,     /*optimizations*/    \
    0u,       /*numOptimizations*/ \
    NULL,     /*sdkBuildId*/       \
    NULL,     /*sdkApiVersion*/    \
    NULL,     /*packageInfo*/      \
    NULL,     /*opsetVersion*/     \
    { 0u }    /*reserved*/         \
  }
// clang-format on

//------------------------------------------------------------------------------
//   API Methods
//------------------------------------------------------------------------------

/**
 * @brief Initialize an Op package library's data structures. This function must be called before
 *        any other library functions. Calling multiple times will result in errors after the first
 *        call. This function can be called again after QnnOpPackage_TerminateFn_t.
 *
 * @param[in] infrastructure Global infrastructure object provided by the backend, for use in all
 *                           operations in the package. This is guaranteed to live at least until
 *                           QnnOpPackage_TerminateFn_t returns, and is safe to cache.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Op package library was successfully initialized.
 *         - QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED: This package library
 *           has already been initialized.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE: Op package initialization failed
 *           due to invalid infrastructure content.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Op package library failed to initialize.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_InitFn_t)(
    QnnOpPackage_GlobalInfrastructure_t infrastructure);

/**
 * @brief Terminate an Op package library, freeing all data structures and invalidating any memory
 *        or handles provided by the library. This function may be called again after a subsequent
 *        call to QnnOpPackage_InitFn_t.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Op package library was successfully terminated.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Op package library termination failed.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_TerminateFn_t)();

/**
 * @brief Retrieve a QnnOpPackage_Info_t struct from an Op package library describing all
 *        operations and optimizations provided by the library.
 *
 * @param[out] info Info object for the library. This pointer shall point to memory owned by the op
 *                  package library and remain valid until QnnOpPackage_TerminateFn_t is called on
 *                  the library. The contents of this struct shall not change before
 *                  QnnOpPackage_TerminateFn_t is called.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Info is fetched successfully.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_INFO: 'info' argument was NULL or invalid.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_GetInfoFn_t)(const QnnOpPackage_Info_t** info);

/**
 * @brief Verifies that this op with the specified config can be successfully executed.
 *
 * @param[in] opConfig Op configuration in question.
 *
 * @note  _inputTensors_ and _outputTensors_ inside opConfig must be fully qualified for
 *        complete validation. However, their unique IDs (_id_) are ignored during validation.
 *
 * @return error code:
 *         - QNN_SUCCESS if validation is successful
 *         - QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE: op config validation failed
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: Validation API not supported
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_ValidateOpConfigFn_t)(Qnn_OpConfig_t opConfig);

/**
 * @brief Create Op implementation with executable content for a given node.
 *
 * @pre The corresponding QnnOpPackage_ValidateOpConfigFn_t should return
 *      QNN_SUCCESS for the supplied node.
 *
 * @param[in] graphInfrastructure Infrastructure for the graph to which the node and kernels
 *                                belong. This memory is guaranteed to live at least until all
 *                                created kernels are freed, and may be safely cached.
 *
 * @param[in] node Node object for which kernels should be created. This node may be freed before
 *                 the created kernels. Neither the node nor it's members should be cached.
 *
 * @param[out] opImpl Op implementation with executable content to compute the operation specified
 *                    by _node_. The Op implementation contents will be freed by the backend with
 *                    QnnOpPackage_FreeOpImplFn_t.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Op implementation is created successfully
 *         - QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE: Failed to create op implementation
 *           due to invalid graph infrastructure content.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: one or more invalid arguments (e.g. NULL)
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_CreateOpImplFn_t)(
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* opImpl);

/**
 * @brief Free the resources associated with Op implementation previously allocated by
 *        QnnOpPackage_CreateOpImplFn_t.
 *
 * @param[in] opImpl Op implementation which should be freed.
 *
 * @return Error code:
 *         - QNN_SUCCESS if Op implementation resources are successfully freed.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: _opImpl_ argument was NULL.
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_FreeOpImplFn_t)(QnnOpPackage_OpImpl_t opImpl);

/**
 * @brief See QnnLog_create() in QnnLog.h for documentation.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_LogInitializeFn_t)(QnnLog_Callback_t callback,
                                                            QnnLog_Level_t maxLogLevel);

/**
 * @brief See QnnLog_setLogLevel() in QnnLog.h for documentation.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_LogSetLevelFn_t)(QnnLog_Level_t maxLogLevel);

/**
 * @brief See QnnLog_free() in QnnLog.h for documentation.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_LogTerminateFn_t)(void);

/**
 * @brief Initialize an op package library and create an op package handle.
 *
 * @param[in] infrastructure Global infrastructure object provided by the backend for use in all
 *                           operations in the package.
 *
 * @param[in] callback Callback to handle op package generated logging messages. NULL represents
 *                     that logging is disabled.
 *
 * @param[in] maxLogLevel Maximum level of messages which the op package will generate.
 *
 * @param[out] opPackage The created op package handle.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Op package was successfully created.
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_PLATFORM: Op package attempted to be created on an
 *           unsupported platform.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: if one or more arguments is invalid.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE: Op package initialization failed due to
 *           invalid infrastructure content.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Op package library failed to initialize.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_CreateFn_t)(
    QnnOpPackage_GlobalInfrastructure_t infrastructure,
    QnnLog_Callback_t callback,
    QnnLog_Level_t maxLogLevel,
    Qnn_OpPackageHandle_t* opPackage);

/**
 * @brief Verifies that this op with the specified config can be successfully executed.
 *
 * @param[in] opPackage An op package handle.
 *
 * @param[in] opConfig Op configuration in question.
 *
 * @note  _inputTensors_ and _outputTensors_ inside opConfig must be fully qualified for complete
 *        validation. However, their unique _id_ and _name_ are ignored during validation.
 *
 * @return error code:
 *         - QNN_SUCCESS No error encountered.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_HANDLE: _opPackage_ is not a valid handle.
 *         - QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE: op config validation failed
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: Validation API not supported
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_ValidateOpConfigHandleFn_t)(
    Qnn_OpPackageHandle_t opPackage, Qnn_OpConfig_t opConfig);

/**
 * @brief Create op implementation with executable content for a given node.
 *
 * @pre The corresponding QnnOpPackage_ValidateOpConfigFn_t should return QNN_SUCCESS for the
 *      supplied node.
 *
 * @param[in] opPackage An op package handle.
 *
 * @param[in] graphInfrastructure Infrastructure for the graph to which the node and kernels belong.
 *                                This memory is guaranteed to live at least until all created
 *                                kernels are freed and may be safely cached.
 *
 * @param[in] node Node object for which kernels should be created. This node may be freed before
 *                 the created kernels. Neither the node nor it's members should be cached.
 *
 * @param[out] opImpl Op implementation with executable content to compute the operation specified
 *                    by _node_. The Op implementation contents will be freed by the backend with
 *                    QnnOpPackage_FreeOpImplFn_t.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_HANDLE: _opPackage_ is not a valid handle.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_INFRASTRUCTURE: Failed to create op implementation
 *           due to invalid graph infrastructure content.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: one or more invalid arguments (e.g. NULL)
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_CreateOpImplHandleFn_t)(
    Qnn_OpPackageHandle_t opPackage,
    QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
    QnnOpPackage_Node_t node,
    QnnOpPackage_OpImpl_t* opImpl);

/**
 * @brief Free the resources associated with Op implementation previously allocated by
 *        QnnOpPackage_CreateOpImplFn_t.
 *
 * @param[in] opPackage An op package handle.
 *
 * @param[in] opImpl Op implementation which should be freed.
 *
 * @return Error code:
 *         - QNN_SUCCESS No error encountered.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_HANDLE: _opPackage_ is not a valid handle.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: _opImpl_ argument was NULL.
 *         - QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE: API not supported.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_FreeOpImplHandleFn_t)(Qnn_OpPackageHandle_t opPackage,
                                                               QnnOpPackage_OpImpl_t opImpl);

/**
 * @brief A function to change the log level for the supplied op package handle.
 *
 * @param[in] opPackage An op package handle.
 *
 * @param[in] maxLogLevel New maximum log level.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_HANDLE: _opPackage_ is not a valid handle.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: if maxLogLevel is not a valid log level.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_LogSetLevelHandleFn_t)(Qnn_OpPackageHandle_t opPackage,
                                                                QnnLog_Level_t maxLogLevel);

/**
 * @brief Free all resources associated with an op package handle.
 *
 * @param[in] Op package handle to be freed.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_HANDLE: _opPackage_ is not a valid handle.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Indicates failure to free op package allocated resources.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_FreeFn_t)(Qnn_OpPackageHandle_t opPackage);

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off
/// QnnOpPackage_ImplementationV1_4_t version initializer macro
#define QNN_OP_PACKAGE_API_VERSION_1_4_0 \
{                                        \
  1u, /*major*/                          \
  4u, /*minor*/                          \
  0u  /*patch*/                          \
}

/**
 * @brief Version 1.4 QNN Op Package Implementation structure.
 *
 *        Contains function pointers for each interface method defined in the
 *        1.4 QNN Op Package API.
 */
typedef struct
{
  QnnOpPackage_InitFn_t             init;
  QnnOpPackage_TerminateFn_t        terminate;
  QnnOpPackage_GetInfoFn_t          getInfo;
  QnnOpPackage_ValidateOpConfigFn_t validateOpConfig;
  QnnOpPackage_CreateOpImplFn_t     createOpImpl;
  QnnOpPackage_FreeOpImplFn_t       freeOpImpl;
  QnnOpPackage_LogInitializeFn_t    logInitialize;
  QnnOpPackage_LogSetLevelFn_t      logSetLevel;
  QnnOpPackage_LogTerminateFn_t     logTerminate;
} QnnOpPackage_ImplementationV1_4_t;

/// QnnOpPackage_ImplementationV1_4_t initializer macro
#define QNN_OP_PACKAGE_IMPLEMENTATION_V1_4_INIT \
  {                                             \
    NULL,     /*init*/                          \
    NULL,     /*terminate*/                     \
    NULL,     /*getInfo*/                       \
    NULL,     /*validateOpConfig*/              \
    NULL,     /*createOpImpl*/                  \
    NULL,     /*freeOpImpl*/                    \
    NULL,     /*logInitialize*/                 \
    NULL,     /*logSetLevel*/                   \
    NULL      /*logTerminate*/                  \
  }
// clang-format on

// clang-format off
/// QnnOpPackage_ImplementationV2_0_t version initializer macro
#define QNN_OP_PACKAGE_API_VERSION_2_0_0 \
{                                        \
  2u, /*major*/                          \
  0u, /*minor*/                          \
  0u  /*patch*/                          \
}

/**
 * @brief Version 2.0 QNN Op Package Implementation structure.
 *
 *        Contains function pointers for each interface method defined in the
 *        2.0 QNN Op Package API.
 */
typedef struct
{
  QnnOpPackage_CreateFn_t                 create;
  QnnOpPackage_GetInfoFn_t                getInfo;
  QnnOpPackage_ValidateOpConfigHandleFn_t validateOpConfig;
  QnnOpPackage_CreateOpImplHandleFn_t     createOpImpl;
  QnnOpPackage_FreeOpImplHandleFn_t       freeOpImpl;
  QnnOpPackage_LogSetLevelHandleFn_t      logSetLevel;
  QnnOpPackage_FreeFn_t                   free;
} QnnOpPackage_ImplementationV2_0_t;

/// QnnOpPackage_ImplementationV2_0_t initializer macro
#define QNN_OP_PACKAGE_IMPLEMENTATION_V2_0_INIT \
  {                                             \
    NULL,     /*create*/                        \
    NULL,     /*getInfo*/                       \
    NULL,     /*validateOpConfig*/              \
    NULL,     /*createOpImpl*/                  \
    NULL,     /*freeOpImpl*/                    \
    NULL,     /*logSetLevel*/                   \
    NULL      /*free*/                          \
  }
// clang-format on

/**
 * @brief Structure which provides the package version and implementation
 *        for a given package. Will be queried by the backend using the
 *        package's implementation provider.
 */
typedef struct {
  /// Version of the QNN Op Package Interface which this package provides.
  /// The Op Package Interface is accessed through correspondingly named implementation.
  Qnn_Version_t interfaceVersion;
  union UNNAMED {
    QnnOpPackage_ImplementationV1_4_t v1_4;
    QnnOpPackage_ImplementationV2_0_t v2_0;
  };
} QnnOpPackage_Interface_t;

/// QnnOpPackage_Interface_t initializer macro
#define QNN_OP_PACKAGE_INTERFACE_INIT                      \
  {                                                        \
    QNN_OP_PACKAGE_API_VERSION_1_4_0, /*interfaceVersion*/ \
    {                                                      \
      QNN_OP_PACKAGE_IMPLEMENTATION_V1_4_INIT /*v1_4*/     \
    }                                                      \
  }

/**
 * @brief A function to retrieve the interface provided by the Op package.
 *        The name of this function is not prescribed by Op Package API, but must
 *        be documented by the package developer and supplied to QNN backend by the client.
 *        See QnnBackend_registerOpPackage().
 *
 * @param[out] interface QNN Op Package interface structure, populated with the version and
 *                       interface methods this Op package provides. Caller to manage the lifetime
 *                       of the pointer, though the contents are to be considered invalid if the op
 *                       package library is terminated/unloaded.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Op package interface is successfully retrieved.
 *         - QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT: _interface_ argument was NULL.
 *         - QNN_OP_PACKAGE_ERROR_GENERAL: Other error occurred.
 */
typedef Qnn_ErrorHandle_t (*QnnOpPackage_InterfaceProvider_t)(QnnOpPackage_Interface_t* interface);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
