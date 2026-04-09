//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 * @file
 * @brief   Common API components
 *
 *          A header which contains common components shared between different
 *          parts of the API, for example, definition of "context" type. This
 *          simplifies the cross-inclusion of headers.
 */

#ifndef QNN_COMMON_H
#define QNN_COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

// Macro controlling visibility of QNN API
#ifndef QNN_API
#define QNN_API
#endif

//! \cond
// Macro to enable processing unnamed unions under struct for documentation purposes
#define UNNAMED
//! \endcond

// Provide values to use for API version.
#define QNN_API_VERSION_MAJOR 2
#define QNN_API_VERSION_MINOR 27
#define QNN_API_VERSION_PATCH 0

/// NULL backend identifier.
#define QNN_BACKEND_ID_NULL 0

/*
 * Identifiers for known backends that may be included into the SDK.
 * These identifiers are defined by each backend in Qnn<backend>Common.h.
 * Identifiers must be unique per backend.
 *
 * - QNN_BACKEND_ID_NULL      0
 * - QNN_BACKEND_ID_REFERENCE 1
 * - QNN_BACKEND_ID_SAVER     2
 * - QNN_BACKEND_ID_CPU       3
 * - QNN_BACKEND_ID_GPU       4
 * - QNN_BACKEND_ID_DSP       5
 * - QNN_BACKEND_ID_HTP       6
 */

/// Global value indicating success
#define QNN_SUCCESS 0

// Error code space assigned to API components
#define QNN_MIN_ERROR_COMMON              1000
#define QNN_MAX_ERROR_COMMON              1999
#define QNN_MIN_ERROR_PROPERTY            2000
#define QNN_MAX_ERROR_PROPERTY            2999
#define QNN_MIN_ERROR_OP_PACKAGE          3000
#define QNN_MAX_ERROR_OP_PACKAGE          3999
#define QNN_MIN_ERROR_BACKEND             4000
#define QNN_MIN_ERROR_BACKEND_SAVER       4950
#define QNN_MAX_ERROR_BACKEND_SAVER       4998
#define QNN_MAX_ERROR_BACKEND             4999
#define QNN_MIN_ERROR_CONTEXT             5000
#define QNN_MAX_ERROR_CONTEXT             5999
#define QNN_MIN_ERROR_GRAPH               6000
#define QNN_MAX_ERROR_GRAPH               6999
#define QNN_MIN_ERROR_TENSOR              7000
#define QNN_MAX_ERROR_TENSOR              7999
#define QNN_MIN_ERROR_MEM                 8000
#define QNN_MAX_ERROR_MEM                 8999
#define QNN_MIN_ERROR_SIGNAL              9000
#define QNN_MAX_ERROR_SIGNAL              9999
#define QNN_MIN_ERROR_ERROR               10000
#define QNN_MAX_ERROR_ERROR               10999
#define QNN_MIN_ERROR_LOG                 11000
#define QNN_MAX_ERROR_LOG                 11999
#define QNN_MIN_ERROR_PROFILE             12000
#define QNN_MAX_ERROR_PROFILE             12999
#define QNN_MIN_ERROR_PERF_INFRASTRUCTURE 13000
#define QNN_MAX_ERROR_PERF_INFRASTRUCTURE 13999
#define QNN_MIN_ERROR_DEVICE              14000
#define QNN_MAX_ERROR_DEVICE              14999
#define QNN_MIN_ERROR_GLOBAL_CONFIG       15000
#define QNN_MAX_ERROR_GLOBAL_CONFIG       15999
// Reserved range for QNN system APIs: 30000-50000
#define QNN_MIN_ERROR_SYSTEM    30000
#define QNN_MAX_ERROR_SYSTEM    49999
#define QNN_MIN_ERROR_INTERFACE 60000
#define QNN_MAX_ERROR_INTERFACE 60999

// Utility macros
#define QNN_PASTE_THREE(a, b, c) a##b##c

/// Simple utility to extract 16-bit error code from 64-bit Qnn_ErrorHandle_t
#define QNN_GET_ERROR_CODE(errorHandle) (errorHandle & 0xFFFF)

//=============================================================================
// Data Types
//=============================================================================

// clang-format off

/**
 * @brief A typedef to indicate QNN API return handle. Return error codes from APIs are to be read
 * out from the least significant 16 bits of the field. The higher order bits are reserved for
 * internal tracking purposes.
 */
typedef uint64_t Qnn_ErrorHandle_t;

/**
 * @brief Definition of the QNN handle type. This handle type is the base type for all other QNN
 * handle types. Handles typically have corresponding create and free API functions.
 */
typedef void* Qnn_Handle_t;

/**
 * @brief Definition of the QNN backend handle. Backend handles are often used as a parent when
 * creating handles other QNN components (e.g. contexts).
 */
typedef Qnn_Handle_t Qnn_BackendHandle_t;

/**
 * @brief Definition of the QNN context handle.
 */
typedef Qnn_Handle_t Qnn_ContextHandle_t;

/**
 * @brief Definition of the QNN device handle.
 */
typedef Qnn_Handle_t Qnn_DeviceHandle_t;

/**
 * @brief Definition of the QNN graph handle. Graph handles cannot be free'd.
 */
typedef Qnn_Handle_t Qnn_GraphHandle_t;

/**
 * @brief Definition of the QNN log handle.
 */
typedef Qnn_Handle_t Qnn_LogHandle_t;

/**
 * @brief Definition of the QNN memory handle.
 */
typedef Qnn_Handle_t Qnn_MemHandle_t;

/**
 * @brief Definition of the QNN profile handle.
 */
typedef Qnn_Handle_t Qnn_ProfileHandle_t;

/**
 * @brief An opaque control object which may be used to control the execution behavior of various
 * QNN functions. A signal object may only be used by one call at a time; if the same signal
 * object is supplied to a second call before the first has terminated, the second call will
 * immediately fail with an error. When the call using a signal returns gracefully, the signal
 * object is made available again.
 */
typedef Qnn_Handle_t Qnn_SignalHandle_t;

// clang-format on

/**
 * @brief An enum which defines error codes commonly used across API components.
 */
typedef enum {
  QNN_COMMON_MIN_ERROR = QNN_MIN_ERROR_COMMON,
  //////////////////////////////////////////

  /// API or feature is not supported by implementation.
  QNN_COMMON_ERROR_NOT_SUPPORTED = QNN_MIN_ERROR_COMMON + 0,
  /// Memory allocation related error.
  QNN_COMMON_ERROR_MEM_ALLOC = QNN_MIN_ERROR_COMMON + 2,
  /// System level error, such as related to platform / OS services
  QNN_COMMON_ERROR_SYSTEM = QNN_MIN_ERROR_COMMON + 3,
  /// Invalid function argument
  QNN_COMMON_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_COMMON + 4,
  /// Illegal operation or sequence of operations
  QNN_COMMON_ERROR_OPERATION_NOT_PERMITTED = QNN_MIN_ERROR_COMMON + 5,
  /// Attempt to use QNN API on an unsupported platform
  QNN_COMMON_ERROR_PLATFORM_NOT_SUPPORTED = QNN_MIN_ERROR_COMMON + 6,
  /// Communication errors with platform / OS service (service is recoverable)
  QNN_COMMON_ERROR_SYSTEM_COMMUNICATION = QNN_MIN_ERROR_COMMON + 7,
  /// Loaded libraries are of incompatible versions
  QNN_COMMON_ERROR_INCOMPATIBLE_BINARIES = QNN_MIN_ERROR_COMMON + 8,
  /// Attempt to reload library already loaded in this process
  QNN_COMMON_ERROR_LOADING_BINARIES = QNN_MIN_ERROR_COMMON + 9,
  /// Resource allocation related error.
  QNN_COMMON_ERROR_RESOURCE_UNAVAILABLE = QNN_MIN_ERROR_COMMON + 10,
  /// Communication errors with platform / OS service (service is non-recoverable)
  QNN_COMMON_ERROR_SYSTEM_COMMUNICATION_FATAL = QNN_MIN_ERROR_COMMON + 11,
  /// General error, which has not been identified as any other error type.
  QNN_COMMON_ERROR_GENERAL = QNN_MIN_ERROR_COMMON + 100,

  //////////////////////////////////////////
  QNN_COMMON_MAX_ERROR = QNN_MAX_ERROR_COMMON,
  // Unused, present to ensure 32 bits.
  QNN_COMMON_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnCommon_Error_t;

//=============================================================================
// Public Functions
//=============================================================================

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_COMMON_H
