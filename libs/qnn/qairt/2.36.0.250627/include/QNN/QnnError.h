//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Error handling API
 *
 *          Requires Backend to be initialized.
 *          Provides means to get detailed error information.
 */

#ifndef QNN_ERROR_H
#define QNN_ERROR_H

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
 * @brief QNN Error API result / error codes.
 */
typedef enum {
  QNN_ERROR_MIN_ERROR = QNN_MIN_ERROR_ERROR,
  ////////////////////////////////////////

  /// Qnn Error success
  QNN_ERROR_NO_ERROR = QNN_SUCCESS,
  /// Invalid function argument
  QNN_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_ERROR + 0,
  /// Unrecognized or invalid error handle
  QNN_ERROR_INVALID_ERROR_HANDLE = QNN_MIN_ERROR_ERROR + 1,
  ////////////////////////////////////////
  QNN_ERROR_MAX_ERROR = QNN_MAX_ERROR_ERROR,
  // Unused, present to ensure 32 bits.
  QNN_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnError_Error_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Query QNN backend for string message describing the error code.
 * Returned message should contain basic information about the nature of the
 * error.
 *
 * @param[in] errorHandle   Error handle to request descriptive message for.
 *
 * @param[out] errorMessage Pointer to a null terminated character array containing the message
 *                          associated with the passed errorHandle. The memory is statically
 *                          owned and should not be freed by the caller. If _errorHandle_
 *                          is not recognized, the pointer _errorMessage_ points to is set to
 *                          nullptr.
 *
 * @return Error code:
 *         - QNN_SUCCESS: error string corresponding to the error handle successfully queried
 *         - QNN_ERROR_INVALID_ARGUMENT: _errorMessage_ is null
 *         - QNN_ERROR_INVALID_ERROR_HANDLE: _errorHandle_ not recognized
 */
QNN_API
Qnn_ErrorHandle_t QnnError_getMessage(Qnn_ErrorHandle_t errorHandle, const char** errorMessage);

/**
 * @brief Query QNN backend for verbose string message describing the error code.
 * Returned message should contain detailed information about the nature of the
 * error.
 *
 * @param[in] errorHandle   Error handle to request descriptive message for.
 *
 * @param[out] errorMessage Pointer to a null terminated character array containing the verbose
 *                          message associated with the passed errorHandle. The memory is
 *                          owned by the backend and only freed when the caller invokes
 *                          QnnError_freeVerboseMessage, passing the same error handle. If
 *                          _errorHandle_ is not recognized, the pointer _errorMessage_ points
 *                          to is set to nullptr.
 *
 * @return Error code:
 *         - QNN_SUCCESS: error string corresponding to the error handle successfully queried
 *         - QNN_ERROR_INVALID_ARGUMENT: _errorMessage_ is null
 *         - QNN_ERROR_INVALID_ERROR_HANDLE: _errorHandle_ not recognized by backend
 */
QNN_API
Qnn_ErrorHandle_t QnnError_getVerboseMessage(Qnn_ErrorHandle_t errorHandle,
                                             const char** errorMessage);

/**
 * @brief Inform QNN backend that the memory associated with the verbose message
 * returned by a previous call to QnnError_getVerboseMessage will no longer be
 * accessed by the caller and may be freed.
 *
 * @param[in] errorMessage Address of character buffer returned in previous call to
 *                          QnnError_getVerboseMessage.
 *
 * @return Error code:
 *         - QNN_SUCCESS: backend acknowledges the caller will no longer access memory
 *           associated with previous call to QnnError_getVerboseMessage
 *         - QNN_ERROR_INVALID_ARGUMENT: _errorMessage_ is null or unrecognized
 */
QNN_API
Qnn_ErrorHandle_t QnnError_freeVerboseMessage(const char* errorMessage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_ERROR_H
