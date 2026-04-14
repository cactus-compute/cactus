//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/** @file
 *  @brief QNN HTP component Performance Infrastructure API
 *
 *         Provides interface to the client to control performance and system
 *         settings of the QNN HTP Accelerator
 */

#ifndef QNN_HTP_PERF_INFRASTRUCTURE_H
#define QNN_HTP_PERF_INFRASTRUCTURE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// max rpc polling time allowed - 9999 us
#define QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME 9999

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN HTP PerfInfrastructure API result / error codes.
 *
 */
typedef enum {
  QNN_HTP_PERF_INFRASTRUCTURE_MIN_ERROR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE,
  ////////////////////////////////////////////////////////////////////////

  QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR                 = QNN_SUCCESS,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_HANDLE_PTR = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 0,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT      = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 1,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED_CONFIG = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 2,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_TRANSPORT          = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 3,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_UNSUPPORTED        = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 4,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_MEM_ALLOC          = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 5,
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_FAILED             = QNN_MIN_ERROR_PERF_INFRASTRUCTURE + 6,

  ////////////////////////////////////////////////////////////////////////
  QNN_HTP_PERF_INFRASTRUCTURE_MAX_ERROR = QNN_MAX_ERROR_PERF_INFRASTRUCTURE,
  /// UNDEFINED value that must not be used by client
  QNN_HTP_PERF_INFRASTRUCTURE_ERROR_UNDEFINED = 0x7fffffff
} QnnHtpPerfInfrastructure_Error_t;

/**
 * @brief Allows client to consider (non-zero value) DCVS enable/disable
 * and option parameters, otherwise (zero value)
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SetDcvsEnable_t;

/**
 * @brief Allows client to start (non-zero value) or stop (zero value)
 * participating in DCVS
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_DcvsEnable_t;

/**
 * @brief Allows client to consider (non-zero value) latency parameter,
 * otherwise (zero value)
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SetSleepLatency_t;

/**
 * @brief Allows client to set up the sleep latency in microseconds
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SleepLatency_t;

/**
 * @brief Allows client to consider (non-zero value) sleep disable
 * parameter, otherwise (zero value)
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SetSleepDisable_t;

/**
 * @brief Allows client to disable sleep or low power modes.
 * Pass a non-zero value to disable sleep in HTP
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SleepDisable_t;

/**
 * @brief Allows client to consider (non-zero value) bus clock
 * params, otherwise (zero value)
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SetBusParams_t;

/**
 * @brief Allows client consider (non-zero value) core clock
 * params, otherwise (zero value)
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_SetCoreParams_t;

/**
 * @brief Allows client to set up the RPC control latency in microseconds
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_RpcControlLatency_t;

/**
 * @brief Allows client to set up the RPC polling time in microseconds
 */
typedef uint32_t QnnHtpPerfInfrastructure_RpcPollingTime_t;

/**
 * @brief Allows client to set up the adaptive polling time in microseconds
 */
typedef uint32_t QnnHtpPerfInfrastructure_AdaptivePollingTime_t;

/**
 * @brief Allows client to set up the HMX timeout interval in microseconds
 */
typedef uint32_t QnnHtpPerfInfrastructure_HmxTimeoutIntervalUs_t;

/**
 * @brief sets the minimum size by which user heap should grow
 * when heap is exhausted. This API is expected to be
 * called only once per backend and has a process wide impact
 *
 * Grow size provided in bytes and defaults to 16MB
 */
typedef uint32_t QnnHtpPerfInfrastructure_MemGrowSize_t;

/**
 * @brief Allows client to set default values for HMX frequency.
 * If enabled 1 HMX vote will scale with DCVS Corner if 0 HMX vote
 * needs to be specified manually.
 *
 */
typedef uint32_t QnnHtpPerfInfrastructure_HmxDefault_Vote_t;

/**
 *  @brief Perf modes to specify clock frequency level within
 *  target voltage corner currently applies only for HMX config.
 */
typedef enum {
  // To select max frequency at target voltage corner.
  QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH = 0,
  // To select min frequency at target voltage corner.
  QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW,
  /// UNKNOWN value that must not be used by client
  QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_ClkPerfMode_t;

/**
 * @brief These are the different voltage corners that can
 * be requested by the client to influence the voting scheme
 * for DCVS
 *
 */
typedef enum {
  /// Maps to HAP_DCVS_VCORNER_DISABLE.
  /// Disable setting up voltage corner
  DCVS_VOLTAGE_CORNER_DISABLE = 0x10,
  /// Maps to HAP_DCVS_VCORNER_SVS2.
  /// Set voltage corner to minimum value supported on platform
  DCVS_VOLTAGE_VCORNER_MIN_VOLTAGE_CORNER = 0x20,
  /// Maps to HAP_DCVS_VCORNER_SVS2.
  /// Set voltage corner to SVS2 value for the platform
  DCVS_VOLTAGE_VCORNER_SVS2 = 0x30,
  /// Maps to HAP_DCVS_VCORNER_SVS.
  /// Set voltage corner to SVS value for the platform
  DCVS_VOLTAGE_VCORNER_SVS = 0x40,
  /// Maps to HAP_DCVS_VCORNER_SVS_PLUS.
  /// Set voltage corner to SVS_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_SVS_PLUS = 0x50,
  /// Maps to HAP_DCVS_VCORNER_NOM.
  /// Set voltage corner to NOMINAL value for the platform
  DCVS_VOLTAGE_VCORNER_NOM = 0x60,
  /// Maps to HAP_DCVS_VCORNER_NOM_PLUS.
  /// Set voltage corner to NOMINAL_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_NOM_PLUS = 0x70,
  /// Maps to HAP_DCVS_VCORNER_TURBO.
  /// Set voltage corner to TURBO value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO = 0x80,
  /// Maps to HAP_DCVS_VCORNER_TURBO_PLUS.
  /// Set voltage corner to TURBO_PLUS value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO_PLUS = 0x90,
  /// Maps to HAP_DCVS_VCORNER_TURBO_L2.
  /// Set voltage corner to TURBO_L2 value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO_L2 = 0x92,
  /// Maps to HAP_DCVS_VCORNER_TURBO_L3.
  /// Set voltage corner to TURBO_L3 value for the platform
  DCVS_VOLTAGE_VCORNER_TURBO_L3 = 0x93,
  /// Maps to HAP_DCVS_VCORNER_MAX.
  /// Set voltage corner to maximum value supported on the platform
  DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER = 0xA0,
  /// UNKNOWN value that must not be used by client
  DCVS_VOLTAGE_VCORNER_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_VoltageCorner_t;

/**
 * @brief These are the expanded voltage corners that can
 * be requested by the client to influence the voting scheme
 * for DCVS
 *
 */
typedef enum {
  /// Maps to HAP_DCVS_EXP_VCORNER_DISABLE.
  /// Disable setting up voltage corner
  DCVS_EXP_VCORNER_DISABLE = 0,
  /// Maps to HAP_DCVS_EXP_VCORNER_MIN.
  /// Set voltage corner to minimum value supported on platform
  DCVS_EXP_VCORNER_MIN = 0x100,
  /// Maps to HAP_DCVS_EXP_VCORNER_LOW_SVS_D2.
  /// Set voltage corner to LOWSVS_D2 value for the platform
  DCVS_EXP_VCORNER_LOW_SVS_D2 = 0x134,
  /// Maps to HAP_DCVS_EXP_VCORNER_LOW_SVS_D1.
  /// Set voltage corner to LOWSVS_D1 value for the platform
  DCVS_EXP_VCORNER_LOW_SVS_D1 = 0x138,
  /// Maps to HAP_DCVS_EXP_VCORNER_LOW_SVS.
  /// Set voltage corner to LOWSVS value for the platform
  DCVS_EXP_VCORNER_LOW_SVS = 0x140,
  /// Maps to HAP_DCVS_EXP_VCORNER_SVS.
  /// Set voltage corner to SVS value for the platform
  DCVS_EXP_VCORNER_SVS = 0x180,
  /// Maps to HAP_DCVS_EXP_VCORNER_SVS_L1.
  /// Set voltage corner to SVS_L1 value for the platform
  DCVS_EXP_VCORNER_SVS_L1 = 0x1C0,
  /// Maps to HAP_DCVS_EXP_VCORNER_NOM.
  /// Set voltage corner to NOM value for the platform
  DCVS_EXP_VCORNER_NOM = 0x200,
  /// Maps to HAP_DCVS_EXP_VCORNER_NOM_L1.
  /// Set voltage corner to NOM_L1 value for the platform
  DCVS_EXP_VCORNER_NOM_L1 = 0x240,
  /// Maps to HAP_DCVS_EXP_VCORNER_TUR.
  /// Set voltage corner to TURBO value for the platform
  DCVS_EXP_VCORNER_TUR = 0x280,
  /// Maps to HAP_DCVS_EXP_VCORNER_TUR_L1.
  /// Set voltage corner to TURBO_L1 value for the platform
  DCVS_EXP_VCORNER_TUR_L1 = 0x2A0,
  /// Maps to HAP_DCVS_EXP_VCORNER_TUR_L2.
  /// Set voltage corner to TURBO_L2 value for the platform
  DCVS_EXP_VCORNER_TUR_L2 = 0x2B0,
  /// Maps to HAP_DCVS_EXP_VCORNER_TUR_L3.
  /// Set voltage corner to TURBO_L3 value for the platform
  DCVS_EXP_VCORNER_TUR_L3 = 0x2C0,
  /// Maps to HAP_DCVS_EXP_VCORNER_MAX.
  /// Selects the maximum voltage corner defined for the chipset
  DCVS_EXP_VCORNER_MAX = 0xFFFF,
  /// UNKNOWN value that must not be used by client
  DCVS_EXP_VCORNER_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_ExpVoltageCorner_t;

/**
 * @brief This enum defines all the possible power mode
 *        that a client can set to influence DCVS mode
 */
typedef enum {
  /// Maps to HAP_DCVS_V2_ADJUST_UP_DOWN.
  /// Allows for DCVS to adjust up and down
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN = 0x1,
  /// Maps to HAP_DCVS_V2_ADJUST_ONLY_UP.
  /// Allows for DCVS to adjust up only
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_ONLY_UP = 0x2,
  /// Maps to HAP_DCVS_V2_POWER_SAVER_MODE.
  /// Higher thresholds for power efficiency
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE = 0x4,
  /// Maps to HAP_DCVS_V2_POWER_SAVER_AGGRESSIVE_MODE.
  /// Higher thresholds for power efficiency with faster ramp down
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_AGGRESSIVE_MODE = 0x8,
  /// Maps to HAP_DCVS_V2_PERFORMANCE_MODE.
  /// Lower thresholds for maximum performance
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE = 0x10,
  /// Maps to HAP_DCVS_V2_DUTY_CYCLE_MODE.
  /// The below value applies only for HVX clients:
  ///  - For streaming class clients:
  ///   - detects periodicity based on HVX usage
  ///   - lowers clocks in the no HVX activity region of each period.
  ///  - For compute class clients:
  ///   - Lowers clocks on no HVX activity detects and brings clocks up on detecting HVX activity
  ///   again.
  ///   - Latency involved in bringing up the clock will be at max 1 to 2 ms.
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE = 0x20,
  /// UNKNOWN value that must not be used by client
  QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_PowerMode_t;

/**
 * @brief This struct provides performance infrastructure configuration
 *        associated with setting up of DcvsV3 which allows to select
 *        bus and core operating corners separately
 */
typedef struct {
  uint32_t contextId;
  QnnHtpPerfInfrastructure_SetDcvsEnable_t setDcvsEnable;
  QnnHtpPerfInfrastructure_DcvsEnable_t dcvsEnable;
  QnnHtpPerfInfrastructure_PowerMode_t powerMode;
  QnnHtpPerfInfrastructure_SetSleepLatency_t setSleepLatency;
  QnnHtpPerfInfrastructure_SleepLatency_t sleepLatency;
  QnnHtpPerfInfrastructure_SetSleepDisable_t setSleepDisable;
  QnnHtpPerfInfrastructure_SleepDisable_t sleepDisable;
  QnnHtpPerfInfrastructure_SetBusParams_t setBusParams;
  QnnHtpPerfInfrastructure_VoltageCorner_t busVoltageCornerMin;
  QnnHtpPerfInfrastructure_VoltageCorner_t busVoltageCornerTarget;
  QnnHtpPerfInfrastructure_VoltageCorner_t busVoltageCornerMax;
  QnnHtpPerfInfrastructure_SetCoreParams_t setCoreParams;
  QnnHtpPerfInfrastructure_VoltageCorner_t coreVoltageCornerMin;
  QnnHtpPerfInfrastructure_VoltageCorner_t coreVoltageCornerTarget;
  QnnHtpPerfInfrastructure_VoltageCorner_t coreVoltageCornerMax;
} QnnHtpPerfInfrastructure_DcvsV3_t;

/**
 * @brief This struct provides performance infrastructure configuration
 *        associated with setting up of hmxv2 which allows to select
 *        hmx corner separately. If hmxPickDefault is 1 all voltage corner
 *        params will be ignored. Ensure to use same contextID as used for
 *        DCVS vote.
 */
typedef struct {
  QnnHtpPerfInfrastructure_HmxDefault_Vote_t hmxPickDefault;
  QnnHtpPerfInfrastructure_ExpVoltageCorner_t hmxVoltageCornerMin;
  QnnHtpPerfInfrastructure_ExpVoltageCorner_t hmxVoltageCornerTarget;
  QnnHtpPerfInfrastructure_ExpVoltageCorner_t hmxVoltageCornerMax;
  QnnHtpPerfInfrastructure_ClkPerfMode_t hmxPerfMode;
} QnnHtpPerfInfrastructure_HmxV2_t;

/**
 * @brief This enum defines all the possible performance
 *        options in Htp Performance Infrastructure that
 *        relate to setting up of power levels
 */
typedef enum {
  /// config enum implies the usage of Dcvs v3
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3 = 1,
  /// config enum implies the usage of rpcControlLatencyConfig struct
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY = 2,
  /// config enum implies the usage of rpcPollingTimeConfig struct
  /// this config is only supported on V69 and later
  /// if enabled, this config is applied to entire process
  /// max allowed is QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_MAX_RPC_POLLING_TIME us
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME = 3,
  /// config HMX timeout interval in us. The HMX is turned off after the set interval
  /// time if no interaction with it after an inference is finished.
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_TIMEOUT_INTERVAL_US = 4,
  /// config HMX V2 voting parameters only on supported chips
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2 = 5,
  /// config enum implies the usage of adaptivePollingTime struct
  /// this config can only be enabled in the RPC polling mode
  /// if enabled, this config is applied to the entire process
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME = 6,
  /// UNKNOWN config option which must not be used
  QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_PowerConfigOption_t;

/**
 * @brief This struct provides performance infrastructure configuration
 *         associated with setting up of power levels
 */
typedef struct {
  QnnHtpPerfInfrastructure_PowerConfigOption_t option;
  union UNNAMED {
    QnnHtpPerfInfrastructure_DcvsV3_t dcvsV3Config;
    QnnHtpPerfInfrastructure_RpcControlLatency_t rpcControlLatencyConfig;
    QnnHtpPerfInfrastructure_RpcPollingTime_t rpcPollingTimeConfig;
    QnnHtpPerfInfrastructure_HmxTimeoutIntervalUs_t hmxTimeoutIntervalUsConfig;
    QnnHtpPerfInfrastructure_HmxV2_t hmxV2Config;
    QnnHtpPerfInfrastructure_AdaptivePollingTime_t adaptivePollingTimeConfig;
  };
} QnnHtpPerfInfrastructure_PowerConfig_t;

/// QnnHtpPerfInfrastructure_PowerConfig_t initializer macro
#define QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIG_INIT                  \
  {                                                                    \
    QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_UNKNOWN, /*config*/ \
    {                                                                  \
      0 /*dcvsV3Config*/                                               \
    }                                                                  \
  }

/**
 * @brief This enum defines all the possible performance
 *        options in Htp Performance Infrastructure that
 *        relate to system memory settings
 */
typedef enum {
  /// sets memory grow size
  QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE = 1,
  /// UNKNOWN config option that must not be used
  QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN = 0x7fffffff
} QnnHtpPerfInfrastructure_MemoryConfigOption_t;

/**
 * @brief Provides performance infrastructure configuration
 *        options that are memory specific
 */
typedef struct {
  QnnHtpPerfInfrastructure_MemoryConfigOption_t option;
  union UNNAMED {
    QnnHtpPerfInfrastructure_MemGrowSize_t memGrowSizeConfig;
  };
} QnnHtpPerfInfrastructure_MemoryConfig_t;

/// QnnHtpPerfInfrastructure_MemoryConfig_t initializer macro
#define QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIG_INIT                  \
  {                                                                     \
    QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_UNKNOWN, /*config*/ \
    {                                                                   \
      0 /*memGrowSizeConfig*/                                           \
    }                                                                   \
  }

//=============================================================================
// API Methods
//=============================================================================

/**
 * @brief This API allows client to create power configuration id that
 *        has to be used to set different performance modes.
 *        Power configuration id has to be destroyed by client when not needed.
 *
 * @param[in] deviceId Hardware Device on which this config id needs to be created.
 *
 * @param[in] coreId Core/NSP on which this config id needs to be created.
 *
 * @param[out] powerConfigId Pointer to power configuration id to be created.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if deviceId/coreId
 *            or power configuration id is NULL
 */
typedef Qnn_ErrorHandle_t (*QnnHtpPerfInfrastructure_CreatePowerConfigIdFn_t)(
    uint32_t deviceId, uint32_t coreId, uint32_t* powerConfigId);

/**
 * @brief This API allows client to destroy power configuration id.
 *
 * @param[in] powerConfigId A power configuration id to be destroyed.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if power configuration
 *            id does not exist
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 */
typedef Qnn_ErrorHandle_t (*QnnHtpPerfInfrastructure_DestroyPowerConfigIdFn_t)(
    uint32_t powerConfigId);

/**
 * @brief This API allows client to set up system power configuration that
 *        will enable different performance modes. This API uses
 *        HAP_power_dcvs_v3_payload struct to config HAP power parameters.
 *        Detailed HAP power parameters description please refer to Hexagon
 *        SDK HAP_power_dcvs_v3_payload documentation.
 *
 * @param[in] powerConfigId A power client id to associate calls to system
 *            power settings. A value of 0 implies NULL power client id
 *            and can override every other setting the user process. To
 *            enable power settings for multiple clients in the same
 *            process, use a non-zero power client id.
 *
 * @param[in] config Pointer to a NULL terminated array
 *            of config option for performance configuration.
 *            NULL is allowed and indicates no config options are provided.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if power configuration
 *            does not exist
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 */
typedef Qnn_ErrorHandle_t (*QnnHtpPerfInfrastructure_SetPowerConfigFn_t)(
    uint32_t powerConfigId, const QnnHtpPerfInfrastructure_PowerConfig_t** config);

/**
 * @brief This API allows clients to set up configuration associated with
 *        system memory on a specific device
 *
 * @param[in] deviceId Hardware Device on which this config needs to be applied.
 *
 * @param[in] coreId Core/NSP on which this config needs to be applied.
 *
 * @param[in] config Pointer to a NULL terminated array
 *            of config option for system memory configuration.
 *            NULL is allowed and indicates no config options are provided.
 *
 * @return Error code
 *         \n QNN_SUCCESS: No error encountered
 *         \n QNN_HTP_PERF_INFRASTRUCTURE_ERROR_INVALID_INPUT if deviceId/coreId
 *            or memory configuration does not exist
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         \n QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 */
typedef Qnn_ErrorHandle_t (*QnnHtpPerfInfrastructure_SetMemoryConfigFn_t)(
    uint32_t deviceId, uint32_t coreId, const QnnHtpPerfInfrastructure_MemoryConfig_t** config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_HTP_PERF_INFRASTRUCTURE_H
