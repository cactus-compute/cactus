//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Memory registration component API.
 *
 *          Requires Backend to be initialized.
 *          Provides means to register externally allocated memory with a backend.
 */

#ifndef QNN_MEM_H
#define QNN_MEM_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================

/// Invalid memory file descriptor value
#define QNN_MEM_INVALID_FD -1

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Mem(ory) API result / error codes.
 */
typedef enum {
  QNN_MEM_MIN_ERROR = QNN_MIN_ERROR_MEM,
  ////////////////////////////////////

  /// Qnn Memory success
  QNN_MEM_NO_ERROR = QNN_SUCCESS,
  /// Backend does not support requested functionality
  QNN_MEM_ERROR_NOT_SUPPORTED = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Invalid function argument
  QNN_MEM_ERROR_INVALID_ARGUMENT = QNN_MIN_ERROR_MEM + 0,
  /// Invalid memory handle
  QNN_MEM_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_MEM + 1,
  /// Provided memory has already been registered
  QNN_MEM_ERROR_ALREADY_REGISTERED = QNN_MIN_ERROR_MEM + 2,
  /// Error in memory mapping
  QNN_MEM_ERROR_MAPPING = QNN_MIN_ERROR_MEM + 3,
  /// Invalid memory shape based on a backend's memory restrictions (e.g. alignment incompatibility)
  QNN_MEM_ERROR_INVALID_SHAPE = QNN_MIN_ERROR_MEM + 4,
  /// Backend does not support requested memory type
  QNN_MEM_ERROR_UNSUPPORTED_MEMTYPE = QNN_MIN_ERROR_MEM + 5,

  ////////////////////////////////////
  QNN_MEM_MAX_ERROR = QNN_MAX_ERROR_MEM,
  // Unused, present to ensure 32 bits.
  QNN_MEM_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnMem_Error_t;

/**
 * @brief A struct which describes the shape of memory
 */
typedef struct {
  /// Number of dimensions
  uint32_t numDim;
  /// Array holding size of each dimension. Size of array is = numDim
  uint32_t* dimSize;
  /// Additional configuration in string, for extensibility. Allowed to be NULL
  const char* shapeConfig;
} Qnn_MemShape_t;

#define SHAPE_CONFIG_DATA_FORMAT_UBWC_RGBA8888 "DATA_FORMAT_UBWC_RGBA8888"
#define SHAPE_CONFIG_DATA_FORMAT_UBWC_NV12     "DATA_FORMAT_UBWC_NV12"
#define SHAPE_CONFIG_DATA_FORMAT_UBWC_NV12_Y   "DATA_FORMAT_UBWC_NV12_Y"
#define SHAPE_CONFIG_DATA_FORMAT_UBWC_NV12_UV  "DATA_FORMAT_UBWC_NV12_UV"

// clang-format off
/// Qnn_MemShape_t initializer macro
#define QNN_MEM_SHAPE_INIT    \
  {                           \
    0u,       /*numDim*/      \
    NULL,     /*dimSize*/     \
    NULL      /*shapeConfig*/ \
  }
// clang-format on

/**
 * @brief An enumeration of memory types which may be used to provide data for a QNN tensor.
 */
typedef enum {
  /// Memory allocated by ION manager. ION memory can only be registered with Backend libraries
  /// when a device supports ION manager.
  QNN_MEM_TYPE_ION = 1,
  /// Memory allocated by a custom backend mechanism.
  QNN_MEM_TYPE_CUSTOM = 2,
  /// Memory allocated by DMA-BUF subsystem.
  QNN_MEM_TYPE_DMA_BUF = 3,
  // Unused, present to ensure 32 bits.
  QNN_MEM_TYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_MemType_t;

/**
 * @brief a struct which includes ION related information
 */
typedef struct {
  /// file descriptor for memory, must be set to QNN_MEM_INVALID_FD if not applicable
  int32_t fd;
} Qnn_MemIonInfo_t;

/// Qnn_MemIonInfo_t initializer macro
#define QNN_MEM_ION_INFO_INIT \
  { QNN_MEM_INVALID_FD /*fd*/ }

/**
 * @brief Definition of custom mem info opaque object. This object type is managed by backend
 * specific APIs obtained by a custom backend mechanism.
 */
typedef void* Qnn_MemInfoCustom_t;

/**
 * @brief a struct which includes DMA-BUF related information
 */
typedef struct {
  /// file descriptor for memory, must be set to QNN_MEM_INVALID_FD if not applicable
  int32_t fd;
  /// data pointer, created by app, using mmap on above file descriptor.
  void* data;
} Qnn_MemDmaBufInfo_t;

/// Qnn_MemDmaBufInfo_t initializer macro
#define QNN_MEM_DMA_BUF_INFO_INIT \
  {                               \
    QNN_MEM_INVALID_FD, /*fd*/    \
        NULL            /*data*/  \
  }

/**
 * @brief A struct which describes memory params
 */
typedef struct {
  /// memory shape
  Qnn_MemShape_t memShape;
  /// memory data type
  Qnn_DataType_t dataType;
  /// memory type
  Qnn_MemType_t memType;

  union UNNAMED {
    Qnn_MemIonInfo_t ionInfo;
    Qnn_MemInfoCustom_t customInfo;
    Qnn_MemDmaBufInfo_t dmaBufInfo;
  };
} Qnn_MemDescriptor_t;

// clang-format off
/// Qnn_MemDescriptor_t initializer macro
#define QNN_MEM_DESCRIPTOR_INIT          \
  {                                      \
    QNN_MEM_SHAPE_INIT,     /*memShape*/ \
    QNN_DATATYPE_UNDEFINED, /*dataType*/ \
    QNN_MEM_TYPE_UNDEFINED, /*memType*/  \
    {                                    \
      QNN_MEM_ION_INFO_INIT /*ionInfo*/  \
    }                                    \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Register existing memory to memory handle.
 *        Used to instruct QNN to use this memory directly.
 *
 * @param[in] context A context handle.
 *
 * @param[in] memDescriptors Array of memory descriptors to be registered.
 *
 * @param[in] numDescriptors Number of memory descriptors in the array.
 *
 * @param[out] memHandles Array of allocated memory handles, length is _numDescriptors_. Same shape
 *                        as _memDescriptors_ (i.e. memHandles[n] corresponds to
 *                        memDescriptors[n]).
 *
 * @note memHandles parameter: Array memory is owned by the client. Array size must be at least
 *       _numDescriptors_*sizeof(Qnn_MemHandle_t). The array will be initialized to NULL by the
 *       backend. Upon failure, no memory will be registered and the _memHandles_ array will remain
 *       NULL.
 *
 * @return Error code:
 *         - QNN_SUCCESS: memory was successfully registered
 *         - QNN_MEM_ERROR_NOT_SUPPORTED: backend does not support this API
 *         - QNN_MEM_ERROR_ALREADY_REGISTERED: memory has already been registered
 *         - QNN_MEM_ERROR_UNSUPPORTED_MEMTYPE: backend does not support a memType specified within
 *           _memDescriptors_
 *         - QNN_MEM_ERROR_MAPPING: failed to map between memory file descriptor and memory address
 *         - QNN_MEM_ERROR_INVALID_ARGUMENT: NULL array ptr or invalid memory descriptor
 *         - QNN_MEM_ERROR_INVALID_SHAPE: backend does not support a memShape specified within
 *           _memDescriptors_
 *         - QNN_MEM_ERROR_INVALID_HANDLE: _context_ is not a valid handle
 */
QNN_API
Qnn_ErrorHandle_t QnnMem_register(Qnn_ContextHandle_t context,
                                  const Qnn_MemDescriptor_t* memDescriptors,
                                  uint32_t numDescriptors,
                                  Qnn_MemHandle_t* memHandles);

/**
 * @brief Deregister a memory handle which was registered via QnnMem_register and invalidates
 *        memHandle for the given backend handle.
 *
 * @param[in] memHandles Array of memory handles to be deregistered.
 *
 * @param[in] numHandles Number of memory handles in the array.
 *
 * @note memHandles parameter: Upon failure, all valid handles within _memHandles_ will still be
 *       de-registered.
 *
 * @return Error code:
 *         - QNN_SUCCESS: memory was successfully de-registered
 *         - QNN_MEM_ERROR_NOT_SUPPORTED: backend does not support this API
 *         - QNN_MEM_ERROR_INVALID_ARGUMENT: _memHandles_ is NULL
 *         - QNN_MEM_ERROR_INVALID_HANDLE: a handle within _memHandles_ is NULL/invalid
 */
QNN_API
Qnn_ErrorHandle_t QnnMem_deRegister(const Qnn_MemHandle_t* memHandles, uint32_t numHandles);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_MEM_H
