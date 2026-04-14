//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Device component API.
 *
 *          This is the top level QNN API component for hardware resource management.
 */

#ifndef QNN_DEVICE_H
#define QNN_DEVICE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

/// Reserved value to select a default device
#define QNN_DEVICE_DEFAULT_DEVICE_ID 0xFFFFFFFF

/// Reserved value to select a default core
#define QNN_DEVICE_DEFAULT_CORE_ID 0xFFFFFFFF

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Device API result / error codes.
 */
typedef enum {
  QNN_DEVICE_MIN_ERROR = QNN_MIN_ERROR_DEVICE,
  ////////////////////////////////////////////
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Memory allocation/deallocation failure
  QNN_DEVICE_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// Invalid function argument
  QNN_DEVICE_ERROR_INVALID_ARGUMENT = QNN_COMMON_ERROR_INVALID_ARGUMENT,
  /// Invalid handle
  QNN_DEVICE_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_DEVICE + 0,
  /// Invalid config values
  QNN_DEVICE_ERROR_INVALID_CONFIG = QNN_MIN_ERROR_DEVICE + 1,
  /// Hardware unavailable
  QNN_DEVICE_ERROR_HARDWARE_UNAVAILABLE = QNN_MIN_ERROR_DEVICE + 2,
  /// Device is associated to a context
  QNN_DEVICE_ERROR_ASSOCIATED_TO_CONTEXT = QNN_MIN_ERROR_DEVICE + 3,
  /// Qnn Device success
  QNN_DEVICE_NO_ERROR = QNN_SUCCESS,

  ////////////////////////////////////////////
  QNN_DEVICE_MAX_ERROR = QNN_MAX_ERROR_DEVICE,
  // Unused, present to ensure 32 bits.
  QNN_DEVICE_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnDevice_Error_t;

/**
 * @brief Backend specific opaque infrastructure object
 *
 * Please refer to the documentation provided by the backend for usage information.
 */
typedef struct _QnnDevice_Infrastructure_t* QnnDevice_Infrastructure_t;

/**
 * @brief Backend-defined structure to populate backend specific information for core info
 */
typedef struct _QnnDevice_CoreInfoExtension_t* QnnDevice_CoreInfoExtension_t;

/**
 * @brief Version 1 of the structure defining per Core info
 */
typedef struct {
  /// ID of the enumerated core
  uint32_t coreId;
  /// Type of the core, as specified by the backend
  uint32_t coreType;
  /// Backend specific extension for core info. Refer to backend headers for the definition
  QnnDevice_CoreInfoExtension_t coreInfoExtension;
} QnnDevice_CoreInfoV1_t;

// clang-format off
/// QnnDevice_CoreInfoV1_t initializer macro
#define QNN_DEVICE_CORE_INFO_V1_INIT                  \
  {                                                   \
    QNN_DEVICE_DEFAULT_CORE_ID, /*coreId*/            \
    0u,                         /*coreType*/          \
    NULL                        /*coreInfoExtension*/ \
  }
// clang-format on

/**
 * @brief Enum to distinguish core info versions
 */
typedef enum {
  QNN_DEVICE_CORE_INFO_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_DEVICE_CORE_INFO_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnDevice_CoreInfoVersion_t;

/**
 * @brief Structure defining per core info
 */
typedef struct {
  QnnDevice_CoreInfoVersion_t version;
  union UNNAMED {
    /// Core info which corresponds to version QNN_DEVICE_CORE_INFO_VERSION_1
    QnnDevice_CoreInfoV1_t v1;
  };
} QnnDevice_CoreInfo_t;

/// QnnDevice_CoreInfo_t initializer macro
#define QNN_DEVICE_CORE_INFO_INIT               \
  {                                             \
    QNN_DEVICE_CORE_INFO_VERSION_1, /*version*/ \
    {                                           \
      QNN_DEVICE_CORE_INFO_V1_INIT /*v1*/       \
    }                                           \
  }

/**
 * @brief Backend-defined structure to populate backend specific information for device info
 */
typedef struct _QnnDevice_DeviceInfoExtension_t* QnnDevice_DeviceInfoExtension_t;

/**
 * @brief Version 1 of the structure defining Hardware Device info
 */
typedef struct {
  /// ID of the device
  uint32_t deviceId;
  /// Type of the device
  uint32_t deviceType;
  /// Number of cores in a device
  uint32_t numCores;
  /// Array of core info structures
  QnnDevice_CoreInfo_t* cores;
  /// Backend specific extension for device info. Refer to backend headers for the definition
  QnnDevice_DeviceInfoExtension_t deviceInfoExtension;
} QnnDevice_HardwareDeviceInfoV1_t;

// clang-format off
/// QnnDevice_HardwareDeviceInfoV1_t initializer macro
#define QNN_DEVICE_HARDWARE_DEVICE_INFO_V1_INIT           \
  {                                                       \
    QNN_DEVICE_DEFAULT_DEVICE_ID, /*deviceId*/            \
    0u,                           /*deviceType*/          \
    0u,                           /*numCores*/            \
    NULL,                         /*cores*/               \
    NULL                          /*deviceInfoExtension*/ \
  }
// clang-format on

/**
 * @brief Enum to distinguish device info versions
 */
typedef enum {
  QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnDevice_HardwareDeviceInfoVersion_t;

/**
 * @brief Structure defining hardware device info (typically a SoC or PCIe extension)
 */
typedef struct {
  QnnDevice_HardwareDeviceInfoVersion_t version;
  union UNNAMED {
    /// Device info which corresponds to version QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1
    QnnDevice_HardwareDeviceInfoV1_t v1;
  };
} QnnDevice_HardwareDeviceInfo_t;

/// QnnDevice_HardwareDeviceInfo_t initializer macro
#define QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT               \
  {                                                        \
    QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1, /*version*/ \
    {                                                      \
      QNN_DEVICE_HARDWARE_DEVICE_INFO_V1_INIT /*v1*/       \
    }                                                      \
  }

/**
 * @brief Version 1 of the structure defining platform info
 */
typedef struct {
  /// Number of devices
  uint32_t numHwDevices;
  /// Array of device info structures
  QnnDevice_HardwareDeviceInfo_t* hwDevices;
} QnnDevice_PlatformInfoV1_t;

// clang-format off
/// QnnDevice_PlatformInfoV1_t initializer macro
#define QNN_DEVICE_PLATFORM_INFO_V1_INIT \
  {                                      \
    0u,      /*numHwDevices*/            \
    NULL     /*hwDevices*/               \
  }
// clang-format on

/**
 * @brief Enum to distinguish platform info versions
 */
typedef enum {
  QNN_DEVICE_PLATFORM_INFO_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_DEVICE_PLATFORM_INFO_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnDevice_PlatformInfoVersion_t;

/**
 * @brief Structure defining the platform info
 */
typedef struct {
  QnnDevice_PlatformInfoVersion_t version;
  union UNNAMED {
    /// Platform info which corresponds to version QNN_DEVICE_PLATFORM_INFO_VERSION_1
    QnnDevice_PlatformInfoV1_t v1;
  };
} QnnDevice_PlatformInfo_t;

/// QnnDevice_PlatformInfo_t initializer macro
#define QNN_DEVICE_PLATFORM_INFO_INIT               \
  {                                                 \
    QNN_DEVICE_PLATFORM_INFO_VERSION_1, /*version*/ \
    {                                               \
      QNN_DEVICE_PLATFORM_INFO_V1_INIT /*v1*/       \
    }                                               \
  }

/**
 * @brief Backend specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnDevice_CustomConfig_t;

/**
 * @brief This enum defines config options to control QnnDevice_Config_t
 */
typedef enum {
  /// sets backend custom options
  QNN_DEVICE_CONFIG_OPTION_CUSTOM = 0,
  /// select QnnDevice_PlatformInfo_t
  QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO = 1,
  /// Unused, present to ensure 32 bits.
  QNN_DEVICE_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnDevice_ConfigOption_t;

/**
 * @brief This struct provides device configuration.
 */
typedef struct {
  QnnDevice_ConfigOption_t option;
  union UNNAMED {
    QnnDevice_CustomConfig_t customConfig;
    QnnDevice_PlatformInfo_t* hardwareInfo;
  };
} QnnDevice_Config_t;

/// QnnDevice_Config_t initializer macro
#define QNN_DEVICE_CONFIG_INIT                     \
  {                                                \
    QNN_DEVICE_CONFIG_OPTION_UNDEFINED, /*option*/ \
    {                                              \
      NULL /*customConfig*/                        \
    }                                              \
  }

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to get the collection of devices and cores that a QNN backend is able to
 *        recognize and communicate with. Memory is owned by the backend and deallocated with a call
 *        to QnnDevice_freePlatformInfo().
 *
 * @note This function may not be supported for offline preparation
 *
 * @param[in] logger A handle to the logger, use NULL handle to disable logging. QnnDevice doesn't
 *                   manage the lifecycle of logger and must be freed by using QnnLog_free().
 *
 * @param[out] platformInfo Information about the platform. Memory for this information is owned
 *                          and managed by QNN backend.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: _platformInfo_ is NULL
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *         - QNN_DEVICE_ERROR_MEM_ALLOC: failure in allocating memory for _platformInfo_
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: invalid _logger_
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_getPlatformInfo(Qnn_LogHandle_t logger,
                                            const QnnDevice_PlatformInfo_t** platformInfo);

/**
 * @brief A function to free the memory allocated during QnnDevice_getPlatformInfo()
 *
 * @note This function may not be supported for offline preparation.
 *
 * @param[in] logger A handle to the logger, use NULL handle to disable logging. QnnDevice doesn't
 *                   manage the lifecycle of logger and must be freed by using QnnLog_free().
 *
 * @param[in] platformInfo Information about the platform. Memory for this information is owned and
 *                         managed by QNN backend.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: _platformInfo_ is NULL
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *         - QNN_DEVICE_ERROR_MEM_ALLOC: failure in de-allocating memory for _platformInfo_
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: invalid _logger_
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_freePlatformInfo(Qnn_LogHandle_t logger,
                                             const QnnDevice_PlatformInfo_t* platformInfo);

/**
 * @brief Get device hardware infrastructure interface object
 *
 * This is optional capability, support is advertised via QnnProperty. If supported, please refer
 * to documentation and/or header file provided by the backend for usage information.
 *
 * @param[out] deviceInfra Pointer to infrastructure interface object. The pointer returned is a
 *                         backend owned memory.
 *
 * @return Error code:
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: _device_ is not a valid handle
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: if _deviceInfra_ is NULL
 *         - QNN_DEVICE_ERROR_MEM_ALLOC: insufficient memory to return _deviceInfra_
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_getInfrastructure(const QnnDevice_Infrastructure_t* deviceInfra);

/**
 * @brief Create a logical device handle to a subset of hardware resources available on the
 *        platform.
 *
 * @param[in] logger A handle to the logger, use NULL handle to disable logging. QnnDevice doesn't
 *                   manage the lifecycle of logger and must be freed by using QnnLog_free().
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used.
 *
 * @note NULL value for config creates a device handle with default configuration. Unless mentioned
 *       in backend specific headers, default configuration would enable all the devices and cores
 *       present on a platform for which a backend can control.
 *
 * @param[out] device A handle to the created device.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: _device_ is NULL
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: _logger_ is not a valid handle
 *         - QNN_DEVICE_ERROR_INVALID_CONFIG: one or more configuration values is invalid
 *         - QNN_DEVICE_ERROR_MEM_ALLOC: failure in allocating memory when creating device
 *         - QNN_DEVICE_ERROR_HARDWARE_UNAVAILABLE: requested hardware resources are unavailable
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_create(Qnn_LogHandle_t logger,
                                   const QnnDevice_Config_t** config,
                                   Qnn_DeviceHandle_t* device);

/**
 * @brief A function to set/modify configuration options on an already created device.
 *        Backends are not required to support this API.
 *
 * @param[in] device A device handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: _device_ is not a valid handle
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: at least one argument is invalid
 *         - QNN_DEVICE_ERROR_INVALID_CONFIG: one or more configuration values is invalid
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *         - QNN_DEVICE_ERROR_ASSOCIATED_TO_CONTEXT: _device_ has associated contexts. Free the
 *           associations before attempting to change the config.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_setConfig(Qnn_DeviceHandle_t device, const QnnDevice_Config_t** config);

/**
 * @brief A function to get platform info associated with a device handle.
 *
 * @param[in] device A device handle.
 *
 * @param[out] platformInfo Information about the platform. Memory for this information is owned
 *                          and managed by QNN backend.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: _device_ is not a valid handle
 *         - QNN_DEVICE_ERROR_INVALID_ARGUMENT: _platformInfo_ is NULL
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_getInfo(Qnn_DeviceHandle_t device,
                                    const QnnDevice_PlatformInfo_t** platformInfo);

/**
 * @brief Free the created device and perform any deallocation of the resources allocated during
 *        device create.
 *
 * @param[in] device A device handle.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_DEVICE_ERROR_INVALID_HANDLE: _device_ is not a valid handle
 *         - QNN_DEVICE_ERROR_MEM_ALLOC: an error is encountered with de-allocation of associated
 *           memory, failure to invalidate handles or other allocated resources
 *         - QNN_DEVICE_ERROR_ASSOCIATED_TO_CONTEXT: One or more contexts associated with the device
 *           handle is not freed
 *         - QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE: API is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnDevice_free(Qnn_DeviceHandle_t device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_DEVICE_H
