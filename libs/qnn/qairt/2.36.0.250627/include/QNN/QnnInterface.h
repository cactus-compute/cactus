//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN Interface API
 *
 *          QNN Interface is an abstraction combining all QNN component APIs. QNN Interface
 *          provides typedef variant of QNN component APIs and API to get QNN interface object(s).
 *          QNN Interface API can coexist with QNN component APIs. Visibility of Interface and
 *          Component APIs is determined by build configuration, specifically by QNN_API and
 *          QNN_INTERFACE macro definitions.
 */

#ifndef QNN_INTERFACE_H
#define QNN_INTERFACE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

// QNN Component API headers
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnError.h"
#include "QnnGlobalConfig.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnMem.h"
#include "QnnProfile.h"
#include "QnnProperty.h"
#include "QnnSignal.h"
#include "QnnTensor.h"

// QNN Op integration headers
#include "QnnOpDef.h"
#include "QnnOpPackage.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

// Macro controlling visibility of QNN Interface API
#ifndef QNN_INTERFACE
#define QNN_INTERFACE
#endif

// Utility macros for version and name construction
#define QNN_INTERFACE_VER_EVAL(major, minor)          QNN_PASTE_THREE(major, _, minor)
#define QNN_INTERFACE_NAME_EVAL(prefix, body, suffix) QNN_PASTE_THREE(prefix, body, suffix)

// Construct interface type name from version, e.g. QnnInterface_ImplementationV0_0_t
#define QNN_INTERFACE_VER_TYPE_EVAL(ver_major, ver_minor) \
  QNN_INTERFACE_NAME_EVAL(                                \
      QnnInterface_ImplementationV, QNN_INTERFACE_VER_EVAL(ver_major, ver_minor), _t)

// Construct interface name from version, e.g. v0_0
#define QNN_INTERFACE_VER_NAME_EVAL(ver_major, ver_minor) \
  QNN_INTERFACE_NAME_EVAL(v, QNN_INTERFACE_VER_EVAL(ver_major, ver_minor), )

// Interface type name for current API version
#define QNN_INTERFACE_VER_TYPE \
  QNN_INTERFACE_VER_TYPE_EVAL(QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR)

// Interface name for current API version
#define QNN_INTERFACE_VER_NAME \
  QNN_INTERFACE_VER_NAME_EVAL(QNN_API_VERSION_MAJOR, QNN_API_VERSION_MINOR)

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Interface API result / error codes
 */
typedef enum {
  QNN_INTERFACE_MIN_ERROR = QNN_MIN_ERROR_INTERFACE,
  ////////////////////////////////////////

  QNN_INTERFACE_NO_ERROR                = QNN_SUCCESS,
  QNN_INTERFACE_ERROR_NOT_SUPPORTED     = QNN_COMMON_ERROR_NOT_SUPPORTED,
  QNN_INTERFACE_ERROR_INVALID_PARAMETER = QNN_COMMON_ERROR_INVALID_ARGUMENT,

  ////////////////////////////////////////
  QNN_INTERFACE_MAX_ERROR = QNN_MAX_ERROR_INTERFACE
} QnnInterface_Error_t;

//
// From QnnProperty.h
//

/** @brief See QnnProperty_hasCapability()*/
typedef Qnn_ErrorHandle_t (*QnnProperty_HasCapabilityFn_t)(QnnProperty_Key_t key);

//
// From QnnGlobalConfig.h
//

/** @brief See QnnConfig_Set()*/
typedef Qnn_ErrorHandle_t (*QnnGlobalConfig_SetFn_t)(const QnnGlobalConfig_t** config);

//
// From QnnBackend.h
//

/** @brief See QnnBackend_create()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_CreateFn_t)(Qnn_LogHandle_t logger,
                                                   const QnnBackend_Config_t** config,
                                                   Qnn_BackendHandle_t* backend);

/** @brief See QnnBackend_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_SetConfigFn_t)(Qnn_BackendHandle_t backend,
                                                      const QnnBackend_Config_t** config);

/** @brief See QnnBackend_getApiVersion()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_GetApiVersionFn_t)(Qnn_ApiVersion_t* pVersion);

/** @brief See QnnBackend_getBuildId()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_GetBuildIdFn_t)(const char** id);

/** @brief See QnnBackend_registerOpPackage()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_RegisterOpPackageFn_t)(Qnn_BackendHandle_t backend,
                                                              const char* packagePath,
                                                              const char* interfaceProvider,
                                                              const char* target);

/** @brief See QnnBackend_getSupportedOperations()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_GetSupportedOperationsFn_t)(
    Qnn_BackendHandle_t backend,
    uint32_t* numOperations,
    const QnnBackend_OperationName_t** operations);

/** @brief See QnnBackend_validateOpConfig()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_ValidateOpConfigFn_t)(Qnn_BackendHandle_t backend,
                                                             Qnn_OpConfig_t opConfig);

/** @brief See QnnBackend_free()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_FreeFn_t)(Qnn_BackendHandle_t backend);

/** @brief See QnnBackend_getProperty()*/
typedef Qnn_ErrorHandle_t (*QnnBackend_GetPropertyFn_t)(Qnn_BackendHandle_t backendHandle,
                                                        QnnBackend_Property_t** properties);

//
// From QnnContext.h
//

/** @brief See QnnContext_create()*/
typedef Qnn_ErrorHandle_t (*QnnContext_CreateFn_t)(Qnn_BackendHandle_t backend,
                                                   Qnn_DeviceHandle_t device,
                                                   const QnnContext_Config_t** config,
                                                   Qnn_ContextHandle_t* context);

/** @brief See QnnContext_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnContext_SetConfigFn_t)(Qnn_ContextHandle_t context,
                                                      const QnnContext_Config_t** config);

/** @brief See QnnContext_getBinarySize()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetBinarySizeFn_t)(
    Qnn_ContextHandle_t context, Qnn_ContextBinarySize_t* binaryBufferSize);

/** @brief See QnnContext_getBinary()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetBinaryFn_t)(Qnn_ContextHandle_t context,
                                                      void* binaryBuffer,
                                                      Qnn_ContextBinarySize_t binaryBufferSize,
                                                      Qnn_ContextBinarySize_t* writtenBufferSize);

/** @brief See QnnContext_createFromBinary()*/
typedef Qnn_ErrorHandle_t (*QnnContext_CreateFromBinaryFn_t)(
    Qnn_BackendHandle_t backend,
    Qnn_DeviceHandle_t device,
    const QnnContext_Config_t** config,
    const void* binaryBuffer,
    Qnn_ContextBinarySize_t binaryBufferSize,
    Qnn_ContextHandle_t* context,
    Qnn_ProfileHandle_t profile);

/** @brief See QnnContext_free()*/
typedef Qnn_ErrorHandle_t (*QnnContext_FreeFn_t)(Qnn_ContextHandle_t context,
                                                 Qnn_ProfileHandle_t profile);

/** @brief See QnnContext_validateBinary()*/
typedef Qnn_ErrorHandle_t (*QnnContext_ValidateBinaryFn_t)(
    Qnn_BackendHandle_t backend,
    Qnn_DeviceHandle_t device,
    const QnnContext_Config_t** config,
    const void* binaryBuffer,
    Qnn_ContextBinarySize_t binaryBufferSize);

/** @brief See QnnContext_createFromBinaryWithSignal()*/
typedef Qnn_ErrorHandle_t (*QnnContext_CreateFromBinaryWithSignalFn_t)(
    Qnn_BackendHandle_t backend,
    Qnn_DeviceHandle_t device,
    const QnnContext_Config_t** config,
    const void* binaryBuffer,
    Qnn_ContextBinarySize_t binaryBufferSize,
    Qnn_ContextHandle_t* context,
    Qnn_ProfileHandle_t profile,
    Qnn_SignalHandle_t signal);

/** @brief See QnnContext_createFromBinaryListAsync()*/
typedef Qnn_ErrorHandle_t (*QnnContext_CreateFromBinaryListAsyncFn_t)(
    Qnn_BackendHandle_t backend,
    Qnn_DeviceHandle_t device,
    const QnnContext_Params_t** contextParams,
    const QnnContext_Config_t** listConfig,
    Qnn_SignalHandle_t signal);

/** @brief See QnnContext_finalize()*/
typedef Qnn_ErrorHandle_t (*QnnContext_FinalizeFn_t)(Qnn_ContextHandle_t context,
                                                     Qnn_ProfileHandle_t profile);

/** @brief See QnnContext_getBinarySectionSize()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetBinarySectionSizeFn_t)(
    Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t graph,
    QnnContext_SectionType_t section,
    Qnn_ContextBinarySize_t* binaryBufferSize);

/** @brief See QnnContext_getBinarySection()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetBinarySectionFn_t)(
    Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t graph,
    QnnContext_SectionType_t section,
    const QnnContext_Buffer_t* binaryBuffer,
    Qnn_ContextBinarySize_t* writtenBufferSize,
    Qnn_ProfileHandle_t profile,
    Qnn_SignalHandle_t signal);
/** @brief See QnnContext_applyBinarySection()*/
typedef Qnn_ErrorHandle_t (*QnnContext_ApplyBinarySectionFn_t)(
    Qnn_ContextHandle_t context,
    Qnn_GraphHandle_t graph,
    QnnContext_SectionType_t section,
    const QnnContext_Buffer_t* binaryBuffer,
    Qnn_ProfileHandle_t profile,
    Qnn_SignalHandle_t signal);

/** @brief See QnnContext_getProperty()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetPropertyFn_t)(Qnn_ContextHandle_t contextHandle,
                                                        QnnContext_Property_t** properties);

/** @brief See QnnContext_getIncrementalBinary()*/
typedef Qnn_ErrorHandle_t (*QnnContext_GetIncrementalBinaryFn_t)(
    Qnn_ContextHandle_t context,
    const void** binaryBuffer,
    Qnn_ContextBinarySize_t* startOffset,
    Qnn_ContextBinarySize_t* writtenBufferSize);

/** @brief See QnnContext_releaseIncrementalBinary()*/
typedef Qnn_ErrorHandle_t (*QnnContext_ReleaseIncrementalBinaryFn_t)(
    Qnn_ContextHandle_t context, const void* binaryBuffer, Qnn_ContextBinarySize_t startOffset);
//
// From QnnGraph.h
//

/** @brief See QnnGraph_create()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_CreateFn_t)(Qnn_ContextHandle_t contextHandle,
                                                 const char* graphName,
                                                 const QnnGraph_Config_t** config,
                                                 Qnn_GraphHandle_t* graphHandle);

/** @brief See QnnGraph_createSubgraph()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_CreateSubgraphFn_t)(Qnn_GraphHandle_t graphHandle,
                                                         const char* graphName,
                                                         Qnn_GraphHandle_t* subgraphHandle);

/** @brief See QnnGraph_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_SetConfigFn_t)(Qnn_GraphHandle_t graphHandle,
                                                    const QnnGraph_Config_t** config);

/** @brief See QnnGraph_getProperty()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_GetPropertyFn_t)(Qnn_GraphHandle_t graphHandle,
                                                      QnnGraph_Property_t** properties);

/** @brief See QnnGraph_addNode()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_AddNodeFn_t)(Qnn_GraphHandle_t graphHandle,
                                                  Qnn_OpConfig_t opConfig);

/** @brief See QnnGraph_finalize()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_FinalizeFn_t)(Qnn_GraphHandle_t graphHandle,
                                                   Qnn_ProfileHandle_t profileHandle,
                                                   Qnn_SignalHandle_t signalHandle);

/** @brief See QnnGraph_retrieve()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_RetrieveFn_t)(Qnn_ContextHandle_t contextHandle,
                                                   const char* graphName,
                                                   Qnn_GraphHandle_t* graphHandle);

/** @brief See QnnGraph_prepareExecutionEnvironment()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_PrepareExecutionEnvironmentFn_t)(
    Qnn_GraphHandle_t graphHandle, QnnGraph_ExecuteEnvironment_t** envs, uint32_t envSize);

/** @brief See QnnGraph_execute()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_ExecuteFn_t)(Qnn_GraphHandle_t graphHandle,
                                                  const Qnn_Tensor_t* inputs,
                                                  uint32_t numInputs,
                                                  Qnn_Tensor_t* outputs,
                                                  uint32_t numOutputs,
                                                  Qnn_ProfileHandle_t profileHandle,
                                                  Qnn_SignalHandle_t signalHandle);

/** @brief See QnnGraph_executeAsync()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_ExecuteAsyncFn_t)(Qnn_GraphHandle_t graphHandle,
                                                       const Qnn_Tensor_t* inputs,
                                                       uint32_t numInputs,
                                                       Qnn_Tensor_t* outputs,
                                                       uint32_t numOutputs,
                                                       Qnn_ProfileHandle_t profileHandle,
                                                       Qnn_SignalHandle_t signalHandle,
                                                       Qnn_NotifyFn_t notifyFn,
                                                       void* notifyParam);

/** @brief See QnnGraph_releaseExecutionEnvironment()*/
typedef Qnn_ErrorHandle_t (*QnnGraph_ReleaseExecutionEnvironmentFn_t)(
    Qnn_GraphHandle_t graphHandle, const QnnGraph_ExecuteEnvironment_t** envs, uint32_t envSize);

//
// From QnnTensor.h
//

/** @brief See QnnTensor_createContextTensor()*/
typedef Qnn_ErrorHandle_t (*QnnTensor_CreateContextTensorFn_t)(Qnn_ContextHandle_t context,
                                                               Qnn_Tensor_t* tensor);

/** @brief See QnnTensor_createGraphTensor()*/
typedef Qnn_ErrorHandle_t (*QnnTensor_CreateGraphTensorFn_t)(Qnn_GraphHandle_t graph,
                                                             Qnn_Tensor_t* tensor);

/** @brief See QnnTensor_updateContextTensor()*/
typedef Qnn_ErrorHandle_t (*QnnTensor_UpdateContextTensorsFn_t)(Qnn_ContextHandle_t context,
                                                                const Qnn_Tensor_t** tensor,
                                                                uint64_t numTensors);

/** @brief See QnnTensor_updateGraphTensor()*/
typedef Qnn_ErrorHandle_t (*QnnTensor_UpdateGraphTensorsFn_t)(Qnn_GraphHandle_t graph,
                                                              const Qnn_Tensor_t** tensor,
                                                              uint64_t numTensors);

//
// From QnnLog.h
//

/** @brief See QnnLog_create()*/
typedef Qnn_ErrorHandle_t (*QnnLog_CreateFn_t)(QnnLog_Callback_t callback,
                                               QnnLog_Level_t maxLogLevel,
                                               Qnn_LogHandle_t* logger);

/** @brief See QnnLog_setLogLevel()*/
typedef Qnn_ErrorHandle_t (*QnnLog_SetLogLevelFn_t)(Qnn_LogHandle_t logger,
                                                    QnnLog_Level_t maxLogLevel);

/** @brief See QnnLog_free()*/
typedef Qnn_ErrorHandle_t (*QnnLog_FreeFn_t)(Qnn_LogHandle_t logger);

//
// From QnnProfile.h
//

/** @brief See QnnProfile_create()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_CreateFn_t)(Qnn_BackendHandle_t backend,
                                                   QnnProfile_Level_t level,
                                                   Qnn_ProfileHandle_t* profile);

/** @brief See QnnProfile_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_SetConfigFn_t)(Qnn_ProfileHandle_t profileHandle,
                                                      const QnnProfile_Config_t** config);

/** @brief See QnnProfile_getEvents()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_GetEventsFn_t)(Qnn_ProfileHandle_t profile,
                                                      const QnnProfile_EventId_t** profileEventIds,
                                                      uint32_t* numEvents);

/** @brief See QnnProfile_getSubEvents()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_GetSubEventsFn_t)(QnnProfile_EventId_t eventId,
                                                         const QnnProfile_EventId_t** subEventIds,
                                                         uint32_t* numSubEvents);

/** @brief See QnnProfile_getEventData()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_GetEventDataFn_t)(QnnProfile_EventId_t eventId,
                                                         QnnProfile_EventData_t* eventData);

/** @brief See QnnProfile_getExtendedEventData()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_GetExtendedEventDataFn_t)(
    QnnProfile_EventId_t eventId, QnnProfile_ExtendedEventData_t* eventData);

/** @brief See QnnProfile_free()*/
typedef Qnn_ErrorHandle_t (*QnnProfile_FreeFn_t)(Qnn_ProfileHandle_t profile);

//
// From QnnMem.h
//

/** @brief See QnnMem_register()*/
typedef Qnn_ErrorHandle_t (*QnnMem_RegisterFn_t)(Qnn_ContextHandle_t context,
                                                 const Qnn_MemDescriptor_t* memDescriptors,
                                                 uint32_t numDescriptors,
                                                 Qnn_MemHandle_t* memHandles);

/** @brief See QnnMem_deRegister()*/
typedef Qnn_ErrorHandle_t (*QnnMem_DeRegisterFn_t)(const Qnn_MemHandle_t* memHandles,
                                                   uint32_t numHandles);

//
// From QnnDevice.h
//

/** @brief See QnnDevice_getPlatformInfo()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_GetPlatformInfoFn_t)(
    Qnn_LogHandle_t logger, const QnnDevice_PlatformInfo_t** platformInfo);

/** @brief See QnnDevice_freePlatformInfo()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_FreePlatformInfoFn_t)(
    Qnn_LogHandle_t logger, const QnnDevice_PlatformInfo_t* platformInfo);

/** @brief See QnnDevice_getInfrastructure()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_GetInfrastructureFn_t)(
    const QnnDevice_Infrastructure_t* deviceInfra);

/** @brief See QnnDevice_create()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_CreateFn_t)(Qnn_LogHandle_t logger,
                                                  const QnnDevice_Config_t** config,
                                                  Qnn_DeviceHandle_t* device);

/** @brief See QnnDevice_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_SetConfigFn_t)(Qnn_DeviceHandle_t device,
                                                     const QnnDevice_Config_t** config);

/** @brief See QnnDevice_getInfo()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_GetInfoFn_t)(Qnn_DeviceHandle_t device,
                                                   const QnnDevice_PlatformInfo_t** platformInfo);

/** @brief See QnnDevice_free()*/
typedef Qnn_ErrorHandle_t (*QnnDevice_FreeFn_t)(Qnn_DeviceHandle_t device);

//
// From QnnSignal.h
//

/** @brief See QnnSignal_create()*/
typedef Qnn_ErrorHandle_t (*QnnSignal_CreateFn_t)(Qnn_BackendHandle_t backend,
                                                  const QnnSignal_Config_t** config,
                                                  Qnn_SignalHandle_t* signal);

/** @brief See QnnSignal_setConfig()*/
typedef Qnn_ErrorHandle_t (*QnnSignal_SetConfigFn_t)(Qnn_SignalHandle_t signal,
                                                     const QnnSignal_Config_t** config);

/** @brief See QnnSignal_trigger()*/
typedef Qnn_ErrorHandle_t (*QnnSignal_TriggerFn_t)(Qnn_SignalHandle_t signal);

/** @brief See QnnSignal_free()*/
typedef Qnn_ErrorHandle_t (*QnnSignal_FreeFn_t)(Qnn_SignalHandle_t signal);

//
// From QnnError.h
//

/** @brief See QnnError_getMessage()*/
typedef Qnn_ErrorHandle_t (*QnnError_GetMessageFn_t)(Qnn_ErrorHandle_t errorHandle,
                                                     const char** errorMessage);
/** @brief See QnnError_getVerboseMessage()*/
typedef Qnn_ErrorHandle_t (*QnnError_GetVerboseMessageFn_t)(Qnn_ErrorHandle_t errorHandle,
                                                            const char** errorMessage);
/** @brief See QnnError_freeVerboseMessage()*/
typedef Qnn_ErrorHandle_t (*QnnError_FreeVerboseMessageFn_t)(const char* errorMessage);

// clang-format off

/**
 * @brief This struct defines Qnn interface specific to version.
 *        Interface functions are allowed to be NULL if not supported/available.
 *
 */
typedef struct {
  QnnProperty_HasCapabilityFn_t             propertyHasCapability;

  QnnBackend_CreateFn_t                     backendCreate;
  QnnBackend_SetConfigFn_t                  backendSetConfig;
  QnnBackend_GetApiVersionFn_t              backendGetApiVersion;
  QnnBackend_GetBuildIdFn_t                 backendGetBuildId;
  QnnBackend_RegisterOpPackageFn_t          backendRegisterOpPackage;
  QnnBackend_GetSupportedOperationsFn_t     backendGetSupportedOperations;
  QnnBackend_ValidateOpConfigFn_t           backendValidateOpConfig;
  QnnBackend_FreeFn_t                       backendFree;

  QnnContext_CreateFn_t                     contextCreate;
  QnnContext_SetConfigFn_t                  contextSetConfig;
  QnnContext_GetBinarySizeFn_t              contextGetBinarySize;
  QnnContext_GetBinaryFn_t                  contextGetBinary;
  QnnContext_CreateFromBinaryFn_t           contextCreateFromBinary;
  QnnContext_FreeFn_t                       contextFree;

  QnnGraph_CreateFn_t                       graphCreate;
  QnnGraph_CreateSubgraphFn_t               graphCreateSubgraph;
  QnnGraph_SetConfigFn_t                    graphSetConfig;
  QnnGraph_AddNodeFn_t                      graphAddNode;
  QnnGraph_FinalizeFn_t                     graphFinalize;
  QnnGraph_RetrieveFn_t                     graphRetrieve;
  QnnGraph_ExecuteFn_t                      graphExecute;
  QnnGraph_ExecuteAsyncFn_t                 graphExecuteAsync;

  QnnTensor_CreateContextTensorFn_t         tensorCreateContextTensor;
  QnnTensor_CreateGraphTensorFn_t           tensorCreateGraphTensor;

  QnnLog_CreateFn_t                         logCreate;
  QnnLog_SetLogLevelFn_t                    logSetLogLevel;
  QnnLog_FreeFn_t                           logFree;

  QnnProfile_CreateFn_t                     profileCreate;
  QnnProfile_SetConfigFn_t                  profileSetConfig;
  QnnProfile_GetEventsFn_t                  profileGetEvents;
  QnnProfile_GetSubEventsFn_t               profileGetSubEvents;
  QnnProfile_GetEventDataFn_t               profileGetEventData;
  QnnProfile_GetExtendedEventDataFn_t       profileGetExtendedEventData;
  QnnProfile_FreeFn_t                       profileFree;

  QnnMem_RegisterFn_t                       memRegister;
  QnnMem_DeRegisterFn_t                     memDeRegister;

  QnnDevice_GetPlatformInfoFn_t             deviceGetPlatformInfo;
  QnnDevice_FreePlatformInfoFn_t            deviceFreePlatformInfo;
  QnnDevice_GetInfrastructureFn_t           deviceGetInfrastructure;
  QnnDevice_CreateFn_t                      deviceCreate;
  QnnDevice_SetConfigFn_t                   deviceSetConfig;
  QnnDevice_GetInfoFn_t                     deviceGetInfo;
  QnnDevice_FreeFn_t                        deviceFree;

  QnnSignal_CreateFn_t                      signalCreate;
  QnnSignal_SetConfigFn_t                   signalSetConfig;
  QnnSignal_TriggerFn_t                     signalTrigger;
  QnnSignal_FreeFn_t                        signalFree;

  QnnError_GetMessageFn_t                   errorGetMessage;
  QnnError_GetVerboseMessageFn_t            errorGetVerboseMessage;
  QnnError_FreeVerboseMessageFn_t           errorFreeVerboseMessage;

  QnnGraph_PrepareExecutionEnvironmentFn_t  graphPrepareExecutionEnvironment;
  QnnGraph_ReleaseExecutionEnvironmentFn_t  graphReleaseExecutionEnvironment;
  QnnGraph_GetPropertyFn_t                  graphGetProperty;

  QnnContext_ValidateBinaryFn_t             contextValidateBinary;
  QnnContext_CreateFromBinaryWithSignalFn_t contextCreateFromBinaryWithSignal;
  QnnContext_CreateFromBinaryListAsyncFn_t  contextCreateFromBinaryListAsync;
  QnnTensor_UpdateGraphTensorsFn_t          tensorUpdateGraphTensors;
  QnnTensor_UpdateContextTensorsFn_t        tensorUpdateContextTensors;
  QnnContext_GetBinarySectionSizeFn_t       contextGetBinarySectionSize;
  QnnContext_GetBinarySectionFn_t           contextGetBinarySection;
  QnnContext_ApplyBinarySectionFn_t         contextApplyBinarySection;
  QnnBackend_GetPropertyFn_t                backendGetProperty;
  QnnContext_GetPropertyFn_t                contextGetProperty;
  QnnContext_GetIncrementalBinaryFn_t       contextGetIncrementalBinary;
  QnnContext_ReleaseIncrementalBinaryFn_t   contextReleaseIncrementalBinary;
  QnnContext_FinalizeFn_t                   contextFinalize;
  QnnGlobalConfig_SetFn_t                   globalConfigSet;
} QNN_INTERFACE_VER_TYPE;

/// QNN_INTERFACE_VER_TYPE initializer macro
#define QNN_INTERFACE_VER_TYPE_INIT { \
  NULL, /*propertyHasCapability*/ \
  NULL, /*backendCreate*/ \
  NULL, /*backendSetConfig*/ \
  NULL, /*backendGetApiVersion*/ \
  NULL, /*backendGetBuildId*/ \
  NULL, /*backendRegisterOpPackage*/ \
  NULL, /*backendGetSupportedOperations*/ \
  NULL, /*backendValidateOpConfig*/ \
  NULL, /*backendFree*/ \
  NULL, /*contextCreate*/ \
  NULL, /*contextSetConfig*/ \
  NULL, /*contextGetBinarySize*/ \
  NULL, /*contextGetBinary*/ \
  NULL, /*contextCreateFromBinary*/ \
  NULL, /*contextFree*/ \
  NULL, /*graphCreate*/ \
  NULL, /*graphCreateSubgraph*/ \
  NULL, /*graphSetConfig*/ \
  NULL, /*graphAddNode*/ \
  NULL, /*graphFinalize*/ \
  NULL, /*graphRetrieve*/ \
  NULL, /*graphExecute*/ \
  NULL, /*graphExecuteAsync*/ \
  NULL, /*tensorCreateContextTensor*/ \
  NULL, /*tensorCreateGraphTensor*/ \
  NULL, /*logCreate*/ \
  NULL, /*logSetLogLevel*/ \
  NULL, /*logFree*/ \
  NULL, /*profileCreate*/ \
  NULL, /*profileSetConfig*/ \
  NULL, /*profileGetEvents*/ \
  NULL, /*profileGetSubEvents*/ \
  NULL, /*profileGetEventData*/ \
  NULL, /*profileGetExtendedEventData*/ \
  NULL, /*profileFree*/ \
  NULL, /*memRegister*/ \
  NULL, /*memDeRegister*/ \
  NULL, /*deviceGetPlatformInfo*/ \
  NULL, /*deviceFreePlatformInfo*/ \
  NULL, /*deviceGetInfrastructure*/ \
  NULL, /*deviceCreate*/ \
  NULL, /*deviceSetConfig*/ \
  NULL, /*deviceGetInfo*/ \
  NULL, /*deviceFree*/ \
  NULL, /*signalCreate*/ \
  NULL, /*signalSetConfig*/ \
  NULL, /*signalTrigger*/ \
  NULL, /*signalFree*/ \
  NULL, /*errorGetMessage*/ \
  NULL, /*errorGetVerboseMessage*/ \
  NULL, /*errorFreeVerboseMessage*/ \
  NULL, /*graphPrepareExecutionEnvironment*/ \
  NULL, /*graphReleaseExecutionEnvironment*/ \
  NULL, /*graphGetProperty*/ \
  NULL, /*contextValidateBinary*/ \
  NULL, /*contextCreateFromBinaryWithSignal*/\
  NULL, /*contextCreateFromBinaryListAsync*/ \
  NULL, /*tensorUpdateGraphTensor*/ \
  NULL, /*tensorUpdateContextTensor*/ \
  NULL, /*contextGetBinarySectionSize*/ \
  NULL, /*contextGetBinarySection*/ \
  NULL, /*contextApplyBinarySection*/ \
  NULL, /*backendGetProperty*/ \
  NULL, /*contextGetProperty*/ \
  NULL, /*contextGetIncrementalProperty*/ \
  NULL, /*contextReleaseIncrementalProperty*/ \
  NULL, /*contextFinalize*/ \
  NULL, /*globalConfigSet*/ \
}

typedef struct {
  /// Backend identifier. See QnnCommon.h for details.
  /// Allowed to be QNN_BACKEND_ID_NULL in case of single backend library, in which case
  /// clients can deduce backend identifier based on library being loaded.
  uint32_t backendId;
  /// Interface provider name. Allowed to be NULL.
  const char* providerName;
  // API version for provided interface
  Qnn_ApiVersion_t apiVersion;
  union UNNAMED {
    // Core interface type and name: e.g. QnnInterface_ImplementationV0_0_t v0_0;
    QNN_INTERFACE_VER_TYPE  QNN_INTERFACE_VER_NAME;
  };
} QnnInterface_t;

/// QnnInterface_t initializer macro
#define QNN_INTERFACE_INIT                                   \
  {                                                          \
    QNN_BACKEND_ID_NULL,      /*backendId*/                  \
    NULL,                     /*providerName*/               \
    QNN_API_VERSION_INIT,     /*apiVersion*/                 \
    {                                                        \
      QNN_INTERFACE_VER_TYPE_INIT /*QNN_INTERFACE_VER_NAME*/ \
    }                                                        \
  }

// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Get list of available interface providers.
 *
 * @param[out] providerList A pointer to an array of available interface providers. The lifetime of
 *                          returned interface object pointers corresponds to the lifetime of the
 *                          provider library. Contents are to be considered invalid if the provider
 *                          library is terminated/unloaded. This function can be called immediately
 *                          after provider library has been loaded.
 *
 * @param[out] numProviders Number of available interface objects in _providerList_.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error.
 *         - QNN_INTERFACE_INVALID_PARAMETER: Invalid parameter was provided.
 *           Either _providerList_ or _numProviders_ was NULL.
 *         - QNN_INTERFACE_ERROR_NOT_SUPPORTED: API not supported.
 */
QNN_INTERFACE
Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList,
                                            uint32_t* numProviders);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_INTERFACE_H
