//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  Tensor component API.
 *
 *          Requires Backend to be initialized.
 *          Tensors have either Context or Graph scope. Tensors created with
 *          Context scope can be used within Graphs that belong to same Context,
 *          but not vice versa. Tensors hold either operation's static/constant
 *          data or input/output activation data.
 */

#ifndef QNN_TENSOR_H
#define QNN_TENSOR_H

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
 * @brief QNN Tensor API result / error codes.
 */
typedef enum {
  QNN_TENSOR_MIN_ERROR = QNN_MIN_ERROR_TENSOR,
  //////////////////////////////////////////

  /// Success.
  QNN_TENSOR_NO_ERROR = QNN_SUCCESS,
  /// Invalid context/graph handle in creating tensor.
  QNN_TENSOR_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_TENSOR + 1,
  /// Tensor with specified credentials not registered with a context/graph.
  QNN_TENSOR_ERROR_DOES_NOT_EXIST = QNN_MIN_ERROR_TENSOR + 2,
  /// (deprecated) Tensor has already been registered with backend.
  QNN_TENSOR_ERROR_ALREADY_EXISTS = QNN_MIN_ERROR_TENSOR + 3,
  /// Invalid tensor param.
  QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM = QNN_MIN_ERROR_TENSOR + 4,
  /// This tensor param is currently unsupported.
  QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM = QNN_MIN_ERROR_TENSOR + 5,
  /// (deprecated) A hash collision has occurred with a previously registered tensor's name.
  QNN_TENSOR_ERROR_NAME_HASH_COLLISION = QNN_MIN_ERROR_TENSOR + 6,
  /// Tensor provided for update is invalid.
  QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE = QNN_MIN_ERROR_TENSOR + 7,
  /// There is optional API component that is not supported yet. See QnnProperty.
  QNN_TENSOR_ERROR_UNSUPPORTED_FEATURE = QNN_COMMON_ERROR_NOT_SUPPORTED,

  //////////////////////////////////////////
  QNN_TENSOR_MAX_ERROR = QNN_MAX_ERROR_TENSOR,
  // Unused, present to ensure 32 bits.
  QNN_TENSOR_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnTensor_Error_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create a new tensor on Qnn_ContextHandle_t.
 *
 *        This call may or may not allocate memory, depending on the Qnn_TensorType_t
 *        value specified in tensor and the accelerator implementation.
 *        Optionally it may be initialized with data provided in the tensor if present.
 *
 * @warning Context tensors cannot be of type QNN_TENSOR_TYPE_NATIVE.
 *          Native tensors connect nodes within a single graph.
 *
 * @warning Context tensors cannot be of datatype QNN_DATATYPE_STRING.
 *
 * @param[in] context The context in which the tensor would be created.
 *
 * @param[in,out] tensor Pointer to a user-allocated struct containing information on the tensor
 *                       (type, name, data format, dimensions, data, etc). For tensors containing
 *                       static data (such as weights or biases), the tensor type is expected to be
 *                       QNN_TENSOR_TYPE_STATIC. Valid data must be presented in the tensor object
 *                       at creation. This data will be copied, and may be safely de-allocated
 *                       after this call returns. Other tensor types (e.g: APP_READ, APP_WRITE,
 *                       APP_READWRITE, NULL) must have the data pointer set to NULL at the time of
 *                       creation. Any preset value in _id_ will be overwritten by the backend as
 *                       part of this call. Subsequent usage of the tensor must reference this _id_.
 *                       Creating a tensor with a name that duplicates a previously created tensor
 *                       name in the context and all child graphs results in undefined behaviour.
 *                       The _dimensions_ are treated as the maximum dimensions during tensor
 *                       creation.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Successfully created a context tensor
 *         - QNN_TENSOR_ERROR_INVALID_HANDLE: Provided context handle is invalid
 *         - QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM: One or more tensor parameters is invalid
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM: One or more tensor parameters are
 *           unsupported
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Failure in creating tensor due to issues with memory
 *           allocation
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnTensor_createContextTensor(Qnn_ContextHandle_t context, Qnn_Tensor_t* tensor);

/**
 * @brief A function to create a new tensor on Qnn_GraphHandle_t.
 *
 *        This call may or may not allocate memory, depending on the Qnn_TensorType_t
 *        value specified in tensor and the accelerator implementation.
 *        Optionally it may be initialized with data provided in the tensor if present.
 *
 * @warning Graph tensors cannot be of type QNN_TENSOR_TYPE_APP_READWRITE. R/W tensors connect
 *          multiple graphs.
 *
 * @warning Graph tensors cannot be of datatype QNN_DATATYPE_STRING.
 *
 * @param[in] graph The graph or sub-graph in which the tensor would be created.
 *
 * @param[in,out] tensor Pointer to a user-allocated struct containing information on the tensor
 *                (type, name, data format, dimensions, data, etc). For tensors containing static
 *                data (such as weights or biases), the tensor type is expected to be
 *                QNN_TENSOR_TYPE_STATIC. Valid data must be presented in the tensor object at
 *                creation. This data will be copied, and may be safely de-allocated after this
 *                call returns. Other tensor types (e.g: NATIVE, APP_READ, APP_WRITE, NULL) must
 *                have the data pointer set to NULL at the time of creation. Any preset value in
 *                _id_ will be overwritten by the backend as part of this call. Subsequent usage of
 *                the tensor must reference this _id_. Creating a tensor with a name that
 *                duplicates a previously created tensor name in the graph or parent context
 *                results in undefined behaviour. The _dimensions_ are treated as the maximum
 *                dimensions during tensor creation.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Successfully created a graph tensor
 *         - QNN_TENSOR_ERROR_INVALID_HANDLE: Provided graph handle is invalid
 *         - QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM: One or more tensor parameters is invalid
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM: One or more tensor parameters are
 *           unsupported
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Failure in creating tensor due to issues with memory
 *           allocation
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnTensor_createGraphTensor(Qnn_GraphHandle_t graph, Qnn_Tensor_t* tensor);

/**
 * @brief Update a graph tensor with the new provided tensor information.
 *        Tensors provided here are associated with the tensor in the backend through the ID field.
 *        Valid fields to update are: data and quantization parameters for UPDATEABLE_STATIC
 *        tensors, quantization parameters for UPDATEABLE_NATIVE, UPDATEABLE_APP_READ,
 *        UPDATEABLE_APP_WRITE, and UPDATEABLE_APP_READWRITE tensors.
 *        Multiple calls to QnnTensor_updateGraphTensors() can be made, but the updates will
 *        not take effect until QnnGraph_finalize() is called.
 *        Backends may support a subset of updateable tensor types.

 *
 *  @return Error code:
 *         - QNN_SUCCESS: Successfully updated the graph tensors
 *         - QNN_TENSOR_ERROR_INVALID_HANDLE: Provided graph handle is invalid
 *         - QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM: One or more tensor parameters is invalid
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM: One or more tensor parameters are
 *           unsupported
 *         - QNN_GRAPH_ERROR_GRAPH_NOT_FINALIZED: graph needs to be finalized before updating
 *           graph tensors.
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Failure in creating tensor due to issues with memory
 *           allocation
 *         - QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE: provided tensor is invalid and cannot
 *           be applied as an update.
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet
 *
 */
QNN_API
Qnn_ErrorHandle_t QnnTensor_updateGraphTensors(Qnn_GraphHandle_t graph,
                                               const Qnn_Tensor_t** tensor,
                                               uint64_t numTensors);

/**
 * @brief Update a context tensor with the new provided tensor information.
 *        Tensors provided here are associated with the tensor in the backend through the ID field.
 *        Valid fields to update are: data and quantization parameters for UPDATEABLE_STATIC
 *        tensors, quantization parameters for UPDATEABLE_NATIVE, UPDATEABLE_APP_READ,
 *        UPDATEABLE_APP_WRITE, and UPDATEABLE_APP_READWRITE tensors. Multiple calls to
 *        QnnTensor_updateContextTensors() can be made, but the updates will not take effect until
 *        QnnGraph_finalize() is called for one or more of the graphs to which the context tensors
 *        are associated. Backends may support a subset of updateable tensor types.
 *
 *  @return Error code:
 *         - QNN_SUCCESS: Successfully updated the context tensor
 *         - QNN_TENSOR_ERROR_INVALID_HANDLE: Provided context handle is invalid
 *         - QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM: One or more tensor parameters is invalid
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_TENSOR_PARAM: One or more tensor parameters are
 *           unsupported
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Failure in creating tensor due to issues with memory
 *           allocation
 *         - QNN_TENSOR_ERROR_INCOMPATIBLE_TENSOR_UPDATE: provided tensor is invalid and cannot
 *           be applied as an update.
 *         - QNN_TENSOR_ERROR_UNSUPPORTED_FEATURE: some API feature is not supported yet
 *
 */
QNN_API
Qnn_ErrorHandle_t QnnTensor_updateContextTensors(Qnn_ContextHandle_t context,
                                                 const Qnn_Tensor_t** tensor,
                                                 uint64_t numTensors);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_TENSOR_H
