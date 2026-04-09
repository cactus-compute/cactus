//=============================================================================
//
//  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Signal component API.
 *
 *          Requires Backend to be initialized.
 *          Provides means to manage Signal objects.
 *          Signal objects are used to control execution of other components.
 */

#ifndef QNN_SIGNAL_H
#define QNN_SIGNAL_H

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
 * @brief QNN Signal API result / error codes.
 */
typedef enum {
  QNN_SIGNAL_MIN_ERROR = QNN_MIN_ERROR_SIGNAL,
  //////////////////////////////////////////

  QNN_SIGNAL_NO_ERROR = QNN_SUCCESS,
  /// Backend does not support the requested functionality
  QNN_SIGNAL_ERROR_UNSUPPORTED = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Attempt to reconfigure, free, or supply to a second QNN function call
  /// a signal object that is already in use.
  QNN_SIGNAL_ERROR_SIGNAL_IN_USE = QNN_MIN_ERROR_SIGNAL + 0,
  /// Signal object is idle and not being used by an outstanding function
  /// call.
  QNN_SIGNAL_ERROR_SIGNAL_IDLE = QNN_MIN_ERROR_SIGNAL + 1,
  /// Invalid configuration error
  QNN_SIGNAL_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_SIGNAL + 2,
  /// NULL or unrecognized signal handle error
  QNN_SIGNAL_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_SIGNAL + 3,
  /// Timeout error
  QNN_SIGNAL_ERROR_TIMEOUT = QNN_MIN_ERROR_SIGNAL + 4,
  /// API supplied with incompatible signal type
  QNN_SIGNAL_ERROR_INCOMPATIBLE_SIGNAL_TYPE = QNN_MIN_ERROR_SIGNAL + 5,
  // Mem allocation error
  QNN_SIGNAL_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,

  //////////////////////////////////////////
  QNN_SIGNAL_MAX_ERROR = QNN_MAX_ERROR_SIGNAL,
  // Unused, present to ensure 32 bits.
  QNN_SIGNAL_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnSignal_Error_t;

/**
 * @brief Custom configuration for Signal object
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnSignal_CustomConfig_t;

/**
 * @brief This enum defines signal config options.
 */
typedef enum {
  /// Sets signal custom options via QnnSignal_CustomConfig_t
  QNN_SIGNAL_CONFIG_OPTION_CUSTOM = 0,
  /// Sets abort on API calls invoked with a signal object.
  /// Abort and Timeout signals are mutually exclusive and
  /// cannot be used together.
  QNN_SIGNAL_CONFIG_OPTION_ABORT = 1,
  /// Sets timeout interval on API calls invoked with a signal
  /// object. Timeout and Abort signals are mutually exclusive
  /// and cannot be used together.
  QNN_SIGNAL_CONFIG_OPTION_TIMEOUT = 2,
  // Unused, present to ensure 32 bits.
  QNN_SIGNAL_CONFIG_UNDEFINED = 0x7FFFFFFF
} QnnSignal_ConfigOption_t;

/**
 * @brief This struct provides signal configuration.
 */
typedef struct {
  /// Type of config object used to configure the signal
  QnnSignal_ConfigOption_t option;
  /// Union of mutually exclusive config values based on
  /// the type specified by 'option'.
  union UNNAMED {
    QnnSignal_CustomConfig_t customConfig;
    /// Timeout interval is represented in microseconds.
    /// Tolerance for the Timeout is platform dependent and
    /// cannot be guaranteed.
    uint64_t timeoutDurationUs;
  };
} QnnSignal_Config_t;

/// QnnSignal_Config_t initializer macro
#define QNN_SIGNAL_CONFIG_INIT              \
  {                                         \
    QNN_SIGNAL_CONFIG_UNDEFINED, /*option*/ \
    {                                       \
      NULL /*customConfig*/                 \
    }                                       \
  }

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Create a new signal object. The object will be configured with desired
 *        behavior and is idle and available for usage.
 *
 * @param[in] backend A backend handle
 *
 * @param[in] config  Pointer to a NULL terminated array of config option pointers.
 *                    NULL is allowed, indicates no config options are provided, and
 *                    signal will not be configured to do anything. All config options
 *                    have default value, in case not provided. If same config
 *                    option type is provided multiple times, the last option value
 *                    will be used.
 *
 * @param[out] signal Handle to newly created signal object.
 *
 * @return Error code:
 *         - QNN_SUCCESS: if the signal is created successfully
 *         - QNN_SIGNAL_ERROR_INVALID_ARGUMENT: at least one argument or config option invalid
 *         - QNN_SIGNAL_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *         - QNN_SIGNAL_ERROR_UNSUPPORTED: if QnnSignal API is not supported on the backend
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 */
QNN_API
Qnn_ErrorHandle_t QnnSignal_create(Qnn_BackendHandle_t backend,
                                   const QnnSignal_Config_t** config,
                                   Qnn_SignalHandle_t* signal);

/**
 * @brief Set/change a configuration on an existing signal
 *
 * @param[in] signal Signal object whose configuration needs to be set
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers.
 *                   NULL is allowed and may be used to reset any previously set configuration.
 *                   No default values are assumed for config options that are not set.
 *                   If same config option type is provided multiple times,
 *                   the last option value will be used. If a backend cannot support
 *                   all provided configs it will fail.
 *
 * @return Error Code:
 *         - QNN_SUCCESS: if the config is set successfully
 *         - QNN_SIGNAL_ERROR_INVALID_HANDLE: signal handle is null or invalid
 *         - QNN_SIGNAL_ERROR_INVALID_ARGUMENT: one or more config values is invalid
 *         - QNN_SIGNAL_ERROR_SIGNAL_IN_USE: when attempting to reconfigure a signal
 *           that is active and in-use.
 *         - QNN_SIGNAL_ERROR_UNSUPPORTED: if QnnSignal API is not supported on the backend
 */
QNN_API
Qnn_ErrorHandle_t QnnSignal_setConfig(Qnn_SignalHandle_t signal, const QnnSignal_Config_t** config);

/**
 * @brief Triggers the signal action during the associated API call. For abort config signals, it
 *        causes the associated API call to gracefully cease execution at the earliest opportunity.
 *        This function will block until the targeted call has released associated resources and is
 *        ready to return in it's own calling context. When the associated API call is initiated,
 *        the signal object will be in-use and not available to another call. When the associated
 *        API call returns, the associated signal object will be available and can safely be passed
 *        to another call.
 *
 * @param[in] signal Signal handle used by the associated API call
 *
 * @return Error code:
 *         - QNN_SUCCESS: if the trigger is successful.
 *         - QNN_SIGNAL_ERROR_INVALID_HANDLE: signal handle is null or invalid
 *         - QNN_SIGNAL_ERROR_INCOMPATIBLE_SIGNAL_TYPE: API does not support the signal type
 *         - QNN_SIGNAL_ERROR_TRIGGER_SIGNAL_IDLE: if the signal is not currently in-use, and hence
 *           can not be triggered.
 *         - QNN_SIGNAL_ERROR_UNSUPPORTED: if QnnSignal API is not supported on the backend
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 */
QNN_API
Qnn_ErrorHandle_t QnnSignal_trigger(Qnn_SignalHandle_t signal);

/**
 * @brief Free memory and resources associated with an available signal object.
 *
 * @param[in] signal The signal object to free.
 *
 * @return Error code:
 *         - QNN_SUCCESS: if the signal object is successfully freed
 *         - QNN_SIGNAL_ERROR_INVALID_HANDLE: signal handle is null or invalid
 *         - QNN_SIGNAL_ERROR_SIGNAL_IN_USE: if the signal object is currently in-use
 *         - QNN_SIGNAL_ERROR_MEM_ALLOC: an error is encountered with de-allocation of associated
 *           memory
 *         - QNN_SIGNAL_ERROR_UNSUPPORTED: if QnnSignal API is not supported on the backend
 */
QNN_API
Qnn_ErrorHandle_t QnnSignal_free(Qnn_SignalHandle_t signal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
