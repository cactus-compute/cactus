//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief  Logging component API.
 *
 *          Provides means for QNN backends to output logging data.
 */

#ifndef QNN_LOG_H
#define QNN_LOG_H

#ifdef __cplusplus
#include <cstdarg>
#else
#include <stdarg.h>
#endif

#include "QnnCommon.h"

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
 * @brief QNN Log API result / error codes.
 */
typedef enum {
  QNN_LOG_MIN_ERROR = QNN_MIN_ERROR_LOG,
  ////////////////////////////////////

  /// Qnn Log success
  QNN_LOG_NO_ERROR = QNN_SUCCESS,
  /// General error relating to memory allocation in Log API
  QNN_LOG_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// Unable to initialize logging
  QNN_LOG_ERROR_INITIALIZATION = QNN_MIN_ERROR_LOG + 2,
  /// Invalid argument passed
  QNN_LOG_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_LOG + 3,
  /// Invalid log handle passed
  QNN_LOG_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_LOG + 4,
  ////////////////////////////////////
  QNN_LOG_MAX_ERROR = QNN_MAX_ERROR_LOG,
  // Unused, present to ensure 32 bits.
  QNN_LOG_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnLog_Error_t;

typedef enum {
  // Enum Levels must be in ascending order, so that the enum value
  // can be compared with the "maximum" set in QnnLog_create().
  QNN_LOG_LEVEL_ERROR   = 1,
  QNN_LOG_LEVEL_WARN    = 2,
  QNN_LOG_LEVEL_INFO    = 3,
  QNN_LOG_LEVEL_VERBOSE = 4,
  /// Reserved for developer debugging
  QNN_LOG_LEVEL_DEBUG = 5,
  // Present to ensure 32 bits
  QNN_LOG_LEVEL_MAX = 0x7fffffff
} QnnLog_Level_t;

/**
 * @brief Signature for user-supplied logging callback.
 *
 * @warning The backend may call this callback from multiple threads, and expects that it is
 *          re-entrant.
 *
 * @param[in] fmt Printf-style message format specifier.
 *
 * @param[in] level Log level for the message. Will not be higher than the maximum specified in
 *                  QnnLog_create.
 *
 * @param[in] timestamp Backend-generated timestamp which is monotonically increasing, but
 *                      otherwise meaningless.
 *
 * @param[in] args Message-specific parameters, to be used with fmt.
 */
typedef void (*QnnLog_Callback_t)(const char* fmt,
                                  QnnLog_Level_t level,
                                  uint64_t timestamp,
                                  va_list args);

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Create a handle to a logger object. This function can be
 *        called before QnnBackend_create().
 *
 * @warning With different logging level enabled, the inference time may vary.
 *
 * @param[in] callback Callback to handle backend-generated logging messages. NULL indicates
 *                     backend may direct log messages to the default log stream on the target
 *                     platform when possible (e.g. to logcat in case of Android).
 *
 * @param[in] maxLogLevel Maximum level of messages which the backend will generate.
 *
 * @param[out] logger The created log handle.
 *
 * @return Error code:
 *         - QNN_SUCCESS: if logging is successfully initialized.
 *         - QNN_COMMON_ERROR_NOT_SUPPORTED: logging is not supported.
 *         - QNN_LOG_ERROR_INVALID_ARGUMENT: if one or more arguments is invalid.
 *         - QNN_LOG_ERROR_MEM_ALLOC: for memory allocation errors.
 *         - QNN_LOG_ERROR_INITIALIZATION: log init failed.
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnLog_create(QnnLog_Callback_t callback,
                                QnnLog_Level_t maxLogLevel,
                                Qnn_LogHandle_t* logger);

/**
 * @brief A function to change the log level for the supplied log handle.
 *
 * @warning With different logging level enabled, the inference time may vary.
 *
 * @param[in] logger A log handle.
 *
 * @param[in] maxLogLevel New maximum log level.
 *
 * @return Error code:
 *         - QNN_SUCCESS: if the level is changed successfully.
 *         - QNN_LOG_ERROR_INVALID_ARGUMENT: if maxLogLevel is not a valid QnnLog_Level_t level.
 *         - QNN_LOG_ERROR_INVALID_HANDLE: _logHandle_ is not a valid handle
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION: SSR occurence (successful recovery)
 *         - QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL: SSR occurence (unsuccessful recovery)
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnLog_setLogLevel(Qnn_LogHandle_t logger, QnnLog_Level_t maxLogLevel);

/**
 * @brief A function to free the memory associated with the log handle.
 *
 * @param[in] logger A log handle.
 *
 * @return Error code:
 *         - QNN_SUCCESS: indicates logging is terminated.
 *         - QNN_LOG_ERROR_MEM_ALLOC: for memory de-allocation errors.
 *         - QNN_LOG_ERROR_INVALID_HANDLE: _logHandle_ is not a valid handle
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnLog_free(Qnn_LogHandle_t logger);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
