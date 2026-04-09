//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/** @file
 *  @brief QNN HTP Device components
 *
 *  This file defines structures and supplements QnnDevice.h for QNN HTP device
 */

#pragma once

#include "QnnCommon.h"
#include "QnnDevice.h"
#include "QnnHtpPerfInfrastructure.h"
#include "QnnTypes.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * This is used to represent the HTP hardware architecture
 * Since QnnDevice only supports V68 or newer, using legacy ARCH will result in error
 */
typedef enum {
  QNN_HTP_DEVICE_ARCH_NONE    = 0,
  QNN_HTP_DEVICE_ARCH_V68     = 68,
  QNN_HTP_DEVICE_ARCH_V69     = 69,
  QNN_HTP_DEVICE_ARCH_V73     = 73,
  QNN_HTP_DEVICE_ARCH_V75     = 75,
  QNN_HTP_DEVICE_ARCH_V79     = 79,
  QNN_HTP_DEVICE_ARCH_V81     = 81,
  QNN_HTP_DEVICE_ARCH_UNKNOWN = 0x7fffffff
} QnnHtpDevice_Arch_t;

/**
 * data struture to configure a device to set the minimum HTP Arch
 * the driver will use ops that compatible to this HTP Arch
 */
typedef struct {
  uint32_t deviceId;
  QnnHtpDevice_Arch_t arch;
} QnnHtpDevice_Minimum_Arch_t;

/**
 * data struture to configure a device to running in Signed/unsigned Domain.
 */
typedef struct {
  uint32_t deviceId;
  bool useSignedProcessDomain;
} QnnHtpDevice_UseSignedProcessDomain_t;

/**
 * data struture to configure a device to running in Secure/normal Domain.
 * running in secure process domain (SecurePD) is only supported in V81 and SecurePD is part of add-on SDK.
 */
typedef struct {
  uint32_t deviceId;
  bool useSecureProcessDomain;
} QnnHtpDevice_UseSecureProcessDomain_t;

/**
 * enum to list what custom configure is available.
 */
typedef enum {
  QNN_HTP_DEVICE_CONFIG_OPTION_SOC      = 0,
  QNN_HTP_DEVICE_CONFIG_OPTION_ARCH     = 1,
  QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD = 2,
  QNN_HTP_DEVICE_CONFIG_OPTION_SECUREPD = 3,
  QNN_HTP_DEVICE_CONFIG_OPTION_UNKNOWN  = 0x7fffffff
} QnnHtpDevice_ConfigOption_t;

/**
 * Data structure for custom configure.
 */
typedef struct {
  QnnHtpDevice_ConfigOption_t option;
  union UNNAMED {
    // This field set the SoC Model
    uint32_t socModel;
    // This field update the minimum HTP arch
    QnnHtpDevice_Minimum_Arch_t arch;
    // This structure is used for enabling/disabling Signed/unsigned PD
    QnnHtpDevice_UseSignedProcessDomain_t useSignedProcessDomain;
    // This structure is used for enabling Secure PD
    QnnHtpDevice_UseSecureProcessDomain_t useSecureProcessDomain;
  };
} QnnHtpDevice_CustomConfig_t;

// For deviceType in QnnDevice_HardwareDeviceInfoV1_t
typedef enum {
  QNN_HTP_DEVICE_TYPE_ON_CHIP = 0,  // HTP cores are inside SoC
  QNN_HTP_DEVICE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_DeviceType_t;

/**
 * @brief QNN HTP Device core type
 * This enumeration provides information about the core type inside the SOC.
 *
 * For online operation, the caller should retrieve this information from
 * `QnnDevice_getPlatformInfo`. For offline operation, the caller needs to create a
 * `QnnDevice_CoreInfo_t` with the correct core type, and then use it to create the
 * `QnnDevice_PlatformInfo_t`.
 */
typedef enum {
  QNN_HTP_CORE_TYPE_NSP   = 0,
  QNN_HTP_CORE_TYPE_HPASS = 1,

  // supported coreType are < QNN_CORE_TYPE_MAX
  QNN_HTP_CORE_TYPE_MAX,
  QNN_HTP_CORE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_CoreType_t;

/**
 * This structure provides info about the NSP device inside SoC
 * For online operation, caller should get these info from QnnDevice_getPlatformInfo
 * For offline operation, caller need to create this structure and filling the correct information
 * for QnnDevice_create
 */
typedef struct {
  size_t vtcmSize;           // The VTCM for this device in Mega Byte
                             // user could not request VTCM size exceed this value
  uint32_t socModel;         // An enum value defined in Qnn Header that represent SoC model
  bool signedPdSupport;      // This field is true if the device supports Signed PD
  bool dlbcSupport;          // This field is true if the device supports DLBC
  QnnHtpDevice_Arch_t arch;  // This field shows the Architecture of this device
} QnnHtpDevice_OnChipDeviceInfoExtension_t;

/**
 * This structure is being used in QnnDevice_HardwareDeviceInfoV1_t
 * QnnDevice_getPlatformInfo use this structure to list the supported device features/info
 */
typedef struct _QnnDevice_DeviceInfoExtension_t {
  QnnHtpDevice_DeviceType_t devType;
  union UNNAMED {
    QnnHtpDevice_OnChipDeviceInfoExtension_t onChipDevice;
  };
} QnnHtpDevice_DeviceInfoExtension_t;

/**
 * @brief QNN HTP Device PerfInfrastructure specialization structure.
 *        Objects of this type are to be referenced through QnnDevice_getInfrastructure.
 *
 *        Contains function pointers for each interface method for
 *        Htp PerfInfrastructure.
 */
typedef struct {
  QnnHtpPerfInfrastructure_CreatePowerConfigIdFn_t createPowerConfigId;
  QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t destroyPowerConfigId;
  QnnHtpPerfInfrastructure_SetPowerConfigFn_t setPowerConfig;
  QnnHtpPerfInfrastructure_SetMemoryConfigFn_t setMemoryConfig;
} QnnHtpDevice_PerfInfrastructure_t;

/// QnnHtpDevice_PerfInfrastructure_t initializer macro
#define QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT \
  {                                             \
    NULL,     /*createPowerConfigId*/           \
        NULL, /*destroyPowerConfigId*/          \
        NULL, /*setPowerConfig*/                \
        NULL  /*setMemoryConfig*/               \
  }

typedef enum {
  QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF    = 0,
  QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_UNKNOWN = 0x7fffffff
} QnnHtpDevice_InfrastructureType_t;

typedef struct _QnnDevice_Infrastructure_t {
  QnnHtpDevice_InfrastructureType_t infraType;
  union UNNAMED {
    QnnHtpDevice_PerfInfrastructure_t perfInfra;
  };
} QnnHtpDevice_Infrastructure_t;

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif
