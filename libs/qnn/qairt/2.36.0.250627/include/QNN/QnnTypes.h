//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which contains the base types required by the API.
 *          Strings are expected to be UTF-8 encoded and NULL terminated.
 */

#ifndef QNN_TYPES_H
#define QNN_TYPES_H

#ifdef __cplusplus
#include <climits>
#include <cstddef>
#include <cstdint>
#else
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================
/**
 * @brief A structure which defines Op Mapping information from Source Framework Operation
 *        to QNN Operation.
 */
typedef enum {
  QNN_OP_MAPPING_TYPE_TENSOR = 0,
  QNN_OP_MAPPING_TYPE_OP = 1
} Qnn_MappingType_t;

typedef struct {
  const char* name;
  Qnn_MappingType_t type;
} Qnn_OpMappingPair_t;

typedef struct {
  /// Name of the QNN Operation or Tensor
  const char* name;
  /// Associated pairs to this tensor or operation.
  Qnn_OpMappingPair_t* pair;
  /// Number of pairs
  uint32_t numPairs;
} Qnn_Mapping_t;

typedef struct {
  const char* graphName;
  Qnn_Mapping_t* opMappings;
  uint32_t numOpMappings;
  Qnn_Mapping_t* tensorMappings;
  uint32_t numTensorMappings;
} Qnn_OpMappingV1_t;

/// Version for Qnn_OpMapping_t
typedef enum {
  QNN_OP_MAPPING_VERSION_1 = 1,
} Qnn_OpMappingVersion_t;

typedef struct {
  Qnn_OpMappingVersion_t version;
  union UNNAMED {
    Qnn_OpMappingV1_t* v1;
  };
} Qnn_OpMapping_t;

// clang-format off
/// Qnn_OpMapping_t initializer macro
#define QNN_OP_MAPPING_INIT        \
  {                                \
    QNN_OP_MAPPING_VERSION_1,      \
    {                              \
      NULL                         \
    }                              \
  }
// clang-format on

// clang-format off
/// Qnn_OpMappingV1_t initializer macro
#define QNN_OP_MAPPING_V1_INIT     \
  {                                \
      NULL, /*graphName*/          \
      NULL, /*opMappings*/         \
      0,    /*numOpMappings*/      \
      NULL, /*tensorMappings*/     \
      0,    /*numTensorMappings*/  \
  }
// clang-format on

/**
 * @brief An enum which defines various data types.
 *
 * @note  4-bit data types (QNN_DATATYPE_SFIXED_POINT_4 and
 *        QNN_DATATYPE_UFIXED_POINT_4) are stored in tightly
 *        packed format into a single byte in little endian
 *        format. This allows two 4-bit quantized elements to be
 *        stored in a single byte. The lower nibble stores the first
 *        value while the higher nibble stores the second value.
 *        For example, to represent two 4-bit quantized values of
 *        10 and 4, they will be stored in a single byte as (0100 1010).
 */
typedef enum {
  // Signed Int: 0x00XX

  /// 8-bit integer type
  QNN_DATATYPE_INT_8 = 0x0008,
  /// 16-bit integer type
  QNN_DATATYPE_INT_16 = 0x0016,
  /// 32-bit integer type
  QNN_DATATYPE_INT_32 = 0x0032,
  /// 64-bit integer type
  QNN_DATATYPE_INT_64 = 0x0064,

  // Unsigned Int: 0x01XX
  QNN_DATATYPE_UINT_8  = 0x0108,
  QNN_DATATYPE_UINT_16 = 0x0116,
  QNN_DATATYPE_UINT_32 = 0x0132,
  QNN_DATATYPE_UINT_64 = 0x0164,

  // Float: 0x02XX
  QNN_DATATYPE_FLOAT_16 = 0x0216,
  QNN_DATATYPE_FLOAT_32 = 0x0232,
  QNN_DATATYPE_FLOAT_64 = 0x0264,

  // Signed Fixed Point: 0x03XX
  QNN_DATATYPE_SFIXED_POINT_4  = 0x0304,
  QNN_DATATYPE_SFIXED_POINT_8  = 0x0308,
  QNN_DATATYPE_SFIXED_POINT_16 = 0x0316,
  QNN_DATATYPE_SFIXED_POINT_32 = 0x0332,

  // Unsigned Fixed Point: 0x04XX
  QNN_DATATYPE_UFIXED_POINT_4  = 0x0404,
  QNN_DATATYPE_UFIXED_POINT_8  = 0x0408,
  QNN_DATATYPE_UFIXED_POINT_16 = 0x0416,
  QNN_DATATYPE_UFIXED_POINT_32 = 0x0432,

  // Bool: 0x05XX
  /// 8-bit boolean type, 0 = false, any non-zero value = true
  QNN_DATATYPE_BOOL_8 = 0x0508,

  // String: 0x06xx
  QNN_DATATYPE_STRING = 0x0608,

  // Unused, present to ensure 32 bits.
  QNN_DATATYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_DataType_t;

/**
 * @brief An enum which defines the different precision modes supported by QNN backends.
 *        A precision mode may be used to express the math type used in the implementation
 *        of an operation.
 */
typedef enum {
  // FLOATING POINT REPRESENTATIONS

  /// 32-bit Floating point precision. The format of the floating point
  /// value is left to backends to choose.
  QNN_PRECISION_FLOAT32 = 0,
  /// 16-bit Floating point precision. The format of the floating point
  /// value is left to backends to choose.
  QNN_PRECISION_FLOAT16 = 1,

  // Unused, present to ensure 32 bits.
  QNN_PRECISION_UNDEFINED = 0x7FFFFFFF
} Qnn_Precision_t;

/**
 * @brief An enum to specify the tensor type, application accessible or native to QNN
 *
 */
typedef enum {
  /// Client application writeable tensor.
  QNN_TENSOR_TYPE_APP_WRITE = 0,
  /// Client application readable tensor.
  QNN_TENSOR_TYPE_APP_READ = 1,
  /// Tensor that can both be read and written by an application. Used in scenarios that may include
  /// supplying an output tensor from one graph as the input to another graph.
  QNN_TENSOR_TYPE_APP_READWRITE = 2,
  /// Tensor native to a graph which may be optimized by a backend and are not accessible by a
  /// client.
  QNN_TENSOR_TYPE_NATIVE = 3,
  /// Static data which doesn't change during execution and may be optimized by a backend. Since the
  /// data cannot change, static tensors cannot have dynamic dimensions.
  QNN_TENSOR_TYPE_STATIC = 4,
  /// Tensor type NULL which can be used to represent optional tensors. Other Qnn_Tensor_t metadata
  /// is ignored.
  QNN_TENSOR_TYPE_NULL = 5,
  /// Tensor containing static data whose content or quantization encodings may
  /// be modified by a client after tensor creation.
  QNN_TENSOR_TYPE_UPDATEABLE_STATIC = 6,
  /// Tensor native to a graph whose quantization encodings may be modified by a
  /// client after tensor creation.
  QNN_TENSOR_TYPE_UPDATEABLE_NATIVE = 7,
  /// Application writable tensor whose quantization encodings may be modified by a
  /// client after tensor creation.
  QNN_TENSOR_TYPE_UPDATEABLE_APP_WRITE = 8,
  /// Application readable tensor whose quantization encodings may be modified by a
  /// client after tensor creation.
  QNN_TENSOR_TYPE_UPDATEABLE_APP_READ = 9,
  /// Application readable/writable tensor whose quantization encodings may be modified by a
  /// client after tensor creation.
  QNN_TENSOR_TYPE_UPDATEABLE_APP_READWRITE = 10,
  /// Tensor type OPTIONAL_APP_WRITE represents an application writable (input) tensor that may be
  /// excluded from inferences
  QNN_TENSOR_TYPE_OPTIONAL_APP_WRITE = 11,
  /// Tensor type OPTIONAL_APP_READ represents an application readable (output) tensor that may be
  /// excluded from inferences.
  QNN_TENSOR_TYPE_OPTIONAL_APP_READ = 12,
  /// Tensor type OPTIONAL_APP_READ_WRITE represents an application readable (output) or writable
  /// (input) tensor that may be excluded from inferences.
  QNN_TENSOR_TYPE_OPTIONAL_APP_READWRITE = 13,
  // Unused, present to ensure 32 bits.
  QNN_TENSOR_TYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorType_t;

/**
 * @brief An enum to specify the parameter type : Scalar or Tensor
 */
typedef enum {
  QNN_PARAMTYPE_SCALAR = 0,
  QNN_PARAMTYPE_TENSOR = 1,
  // Unused, present to ensure 32 bits.
  QNN_PARAMTYPE_UNDEFINED = 0xFFFFFFFF
} Qnn_ParamType_t;

/**
 * @brief An enum to specify definition source for field(s) following this enum
 */
typedef enum {
  /// Indicates backend implementation to update or decide
  QNN_DEFINITION_IMPL_GENERATED = 0,
  /// Indicates that provided definition needs to be used
  QNN_DEFINITION_DEFINED = 1,
  // Unused, present to ensure 32 bits.
  QNN_DEFINITION_UNDEFINED = 0x7FFFFFFF
} Qnn_Definition_t;

/**
 * @brief An enum to specify a priority.
 */
typedef enum {
  /// QNN_PRIORITY_LOW is always available for use.
  QNN_PRIORITY_LOW = 0,
  /// QNN_PRIORITY_NORMAL is always available for use.
  QNN_PRIORITY_NORMAL  = 100,
  QNN_PRIORITY_DEFAULT = QNN_PRIORITY_NORMAL,
  /// QNN_PRIORITY_NORMAL_HIGH usage may be restricted and would silently be treated as
  /// QNN_PRIORITY_NORMAL
  QNN_PRIORITY_NORMAL_HIGH = 150,
  /// QNN_PRIORITY_HIGH usage may be restricted and would silently be treated as
  /// QNN_PRIORITY_NORMAL
  QNN_PRIORITY_HIGH = 200,
  // Unused, present to ensure 32 bits.
  QNN_PRIORITY_UNDEFINED = 0x7FFFFFFF
} Qnn_Priority_t;

/**
 * @brief A typedef to indicate context binary size.
 */
typedef uint64_t Qnn_ContextBinarySize_t;

/**
 * @brief An enum to describe reporting levels for the error handling API
 * QNN_ERROR_REPORTING_LEVEL_BRIEF: get basic information about an error
 * QNN_ERROR_REPORTING_LEVEL_DETAILED: get detailed information about an error
 * in memory-based object forms
 */
typedef enum {
  QNN_ERROR_REPORTING_LEVEL_BRIEF    = 0,
  QNN_ERROR_REPORTING_LEVEL_DETAILED = 1,
  // Unused, present to ensure 32 bits.
  QNN_ERROR_REPORTING_LEVEL_UNDEFINED = 0x7FFFFFFF
} Qnn_ErrorReportingLevel_t;

/**
 * @brief A typedef describing error reporting configuration
 */
typedef struct {
  /// Error reporting level
  Qnn_ErrorReportingLevel_t reportingLevel;
  /// Amount of memory to be reserved for error information. Specified in KB
  uint32_t storageLimit;
} Qnn_ErrorReportingConfig_t;

// clang-format off
/// Qnn_ErrorReportingConfig_t initializer macro
#define QNN_ERROR_REPORTING_CONFIG_INIT                     \
  {                                                         \
    QNN_ERROR_REPORTING_LEVEL_UNDEFINED, /*reportingLevel*/ \
    0u                                   /*storageLimit*/   \
  }
// clang-format on

/**
 * @brief A struct which is used to provide a version number using 3 values:
 * major, minor, patch
 */
typedef struct {
  uint32_t major;
  uint32_t minor;
  uint32_t patch;
} Qnn_Version_t;

// clang-format off
/// Qnn_Version_t initializer macro
#define QNN_VERSION_INIT \
  {                      \
    0u,    /*major*/     \
    0u,    /*minor*/     \
    0u     /*patch*/     \
  }
// clang-format on

/**
 * @brief A struct used to provide the versions of both the core QNN API
 * and any Backend Specific API
 */
typedef struct {
  /// Version of the QNN core API common to all backends
  Qnn_Version_t coreApiVersion;
  /// Version of the backend-specific API
  Qnn_Version_t backendApiVersion;
} Qnn_ApiVersion_t;

/// Qnn_ApiVersion_t initializer macro
#define QNN_API_VERSION_INIT                            \
  {                                                     \
    {                                                   \
        QNN_API_VERSION_MAJOR, /*coreApiVersion.major*/ \
        QNN_API_VERSION_MINOR, /*coreApiVersion.minor*/ \
        QNN_API_VERSION_PATCH  /*coreApiVersion.patch*/ \
    },                                                  \
        QNN_VERSION_INIT /*backendApiVersion*/          \
  }

/**
 * @brief A value representing an immutable value which configures a node.
 */
typedef struct {
  Qnn_DataType_t dataType;
  union UNNAMED {
    float floatValue;
    double doubleValue;
    uint64_t uint64Value;
    int64_t int64Value;
    uint32_t uint32Value;
    int32_t int32Value;
    uint16_t uint16Value;
    int16_t int16Value;
    uint8_t uint8Value;
    int8_t int8Value;
    uint8_t bool8Value;
    const char* stringValue;
  };
} Qnn_Scalar_t;

/// Qnn_Scalar_t initializer macro
#define QNN_SCALAR_INIT                  \
  {                                      \
    QNN_DATATYPE_UNDEFINED, /*dataType*/ \
    {                                    \
      0.0f /*floatValue*/                \
    }                                    \
  }

/**
 * @brief An enum to specify quantization encoding type structure
 *
 */
typedef enum {
  /// Indicates per-tensor scale-offset encoding type. See Qnn_ScaleOffset_t. Support can be checked
  /// via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_SCALE_OFFSET.
  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET = 0,
  /// Indicates per-axis (e.g. per-channel) scale-offset encoding type. See Qnn_AxisScaleOffset_t.
  /// Support can be checked via
  /// QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET.
  QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET = 1,
  /// Indicates bit-width scale-offset encoding type. See Qnn_BwScaleOffset_t. Support can be
  /// checked via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BW_SCALE_OFFSET.
  QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET = 2,
  /// Indicates bit-width per-axis scale-offset encoding type. See Qnn_BwAxisScaleOffset_t. Support
  /// can be checked via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET.
  QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET = 3,
  /// Indicates per-block scale-offset encoding type. See Qnn_BlockScaleOffset_t. Support can be
  /// checked via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCK_SCALE_OFFSET.
  QNN_QUANTIZATION_ENCODING_BLOCK = 4,
  /// Indicates per-block scale-offset encoding type. See Qnn_BlockScaleOffset_t. Support can be
  /// checked via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION.
  QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION = 5,
  /// Indicates VQ compression encoding type. See Qnn_VectorQuantCompression_t. Support can be
  /// checked via QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_VQ_COMPRESSION.
  QNN_QUANTIZATION_ENCODING_VECTOR = 6,
  // Unused, present to ensure 32 bits.
  QNN_QUANTIZATION_ENCODING_UNDEFINED = 0x7FFFFFFF
} Qnn_QuantizationEncoding_t;

/**
 * @brief A struct to express scale-offset quantization encoding.
 *
 * float_value = (quantized_value + offset) * scale
 */
typedef struct {
  /// scale must be strictly positive
  float scale;
  int32_t offset;
} Qnn_ScaleOffset_t;

// clang-format off
/// Qnn_ScaleOffset_t initializer macro
#define QNN_SCALE_OFFSET_INIT \
  {                           \
    0.0f, /*scale*/           \
    0     /*offset*/          \
  }
// clang-format on

/**
 * @brief A struct to express quantization parameters as a positive scale with a zero offset and a
 * bitwidth.
 *
 * float_value = (quantized_value + offset) * scale
 *
 * bitwidth must be > 0, and is used to express the true number of bits used to quantize the value,
 * which may be different from the bitwidth of the tensor indicated by its data type. For example:
 * the quantization encoding for a tensor of type QNN_DATATYPE_UFIXED_POINT_8 that is quantized to
 * 4-bit precision may be expressed by setting bitwidth = 4. In such circumstances, data quantized
 * to a lower precision will still occupy the full extent of bits allotted to the tensor as per its
 * data type in unpacked form.
 *
 * The datatype used must be the smallest type which can accommodate the bitwidth. For example: a
 * tensor quantized to 4-bit precision must use an 8-bit datatype, 16-bit or larger datatypes are
 * not permitted.
 *
 * Tensor elements are expected to occupy the least significant bits of the total size alloted to
 * the datatype, and all bits above the specified bitwidth will be ignored. For example: an 8-bit
 * datatype tensor quantized to 4-bit precision will be interpreted as a 4-bit value contained in
 * the lower 4 bits of each element, and the upper 4 bits will be ignored. For signed datatypes, the
 * value will be interpreted as a two's complement integer where the signed bit is the most
 * significant bit permitted by the specified bitwidth. For example: -3 would be represented as
 * 0b11111101 as a signed 8-bit integer, but can also be represented as 0b00001101 as a signed 4-bit
 * integer stored in an 8-bit container. Either of these representations are valid to express -3 as
 * a 4-bit signed integer in an 8-bit container, and will be treated identically because the upper 4
 * bits will be ignored.
 */
typedef struct {
  /// bitwidth must be <= number of bits specified by data type of tensor
  uint32_t bitwidth;
  /// scale must be strictly positive
  float scale;
  int32_t offset;
} Qnn_BwScaleOffset_t;

// clang-format off
/// Qnn_BwScaleOffset_t initializer macro
#define QNN_BW_SCALE_OFFSET_INIT \
  {                              \
    0u,   /*bitwidth*/           \
    0.0f, /*scale*/              \
    0     /*offset*/             \
  }
// clang-format on

/**
 * @brief A struct to express per-axis quantization parameters as a scale with a zero offset
 */
typedef struct {
  int32_t axis;
  uint32_t numScaleOffsets;
  Qnn_ScaleOffset_t* scaleOffset;
} Qnn_AxisScaleOffset_t;

// clang-format off
/// Qnn_AxisScaleOffset_t initializer macro
#define QNN_AXIS_SCALE_OFFSET_INIT \
  {                                \
    0,       /*axis*/              \
    0u,      /*numScaleOffsets*/   \
    NULL     /*scaleOffset*/       \
  }                                \
// clang-format on

/**
 * @brief A struct to express per-axis quantization parameters as collection of scales, offsets
 * and bitwidth.
 *
 * bitwidth must be > 0 and applies commonly to all axes. It is used to express the true number of
 * bits used to quantize the value, which may be different from the bitwidth of the tensor indicated
 * by its data type. For example: the quantization encoding for a tensor of type
 * QNN_DATATYPE_UFIXED_POINT_8 that is quantized to 4-bit precision may be expressed by setting
 * bitwidth = 4. In such circumstances, data quantized to a lower precision will still occupy the
 * full extent of bits allotted to the tensor as per its data type in unpacked form.
 *
 * The datatype used must be the smallest type which can accommodate the bitwidth. For example: a
 * tensor quantized to 4-bit precision must use an 8-bit datatype, 16-bit or larger datatypes are
 * not permitted.
 *
 * Tensor elements are expected to occupy the least significant bits of the total size alloted to
 * the datatype, and all bits above the specified bitwidth will be ignored. For example: an 8-bit
 * datatype tensor quantized to 4-bit precision will be interpreted as a 4-bit value contained in
 * the lower 4 bits of each element, and the upper 4 bits will be ignored. For signed datatypes, the
 * value will be interpreted as a two's complement integer where the signed bit is the most
 * significant bit permitted by the specified bitwidth. For example: -3 would be represented as
 * 0b11111101 as a signed 8-bit integer, but can also be represented as 0b00001101 as a signed 4-bit
 * integer stored in an 8-bit container. Either of these representations are valid to express -3 as
 * a 4-bit signed integer in an 8-bit container, and will be treated identically because the upper 4
 * bits will be ignored.
 */
typedef struct {
  /// bitwidth must be <= number of bits specified by data type of tensor
  uint32_t bitwidth;
  int32_t axis;
  /// numElements applies to both scales and offsets and they are supposed to be a one-to-one match
  uint32_t numElements;
  /// scales must be strictly positive
  float* scales;
  /// offsets must match scales in their dimension except when it can be NULL to indicate that the
  /// value is symmetrically quantized and hence, offset = 0
  int32_t* offsets;
} Qnn_BwAxisScaleOffset_t;

// clang-format off
/// Qnn_BwAxisScaleOffset_t initializer macro
#define QNN_BW_AXIS_SCALE_OFFSET_INIT \
  {                                   \
    0u,      /*bitwidth*/             \
    0,       /*axis*/                 \
    0u,      /*numElements*/          \
    NULL,    /*scales*/               \
    NULL     /*offsets*/              \
  }
// clang-format on

/**
 * @brief A struct to express block quantization parameters. A tensor is divided into blocks of
 * size blockSize, where blockSize is an array of length rank.
 *
 * @note num of scaleOffsets (i.e. num of blocks) must be ==
 * ceil(dimensions[0]/blockSize[0])*ceil(dimensions[1]/blockSize[1]) ...
 * .... *ceil(dimensions[rank-1] / blockSize[rank-1]). *
 */
typedef struct {
  /// Dimensions of the block in number of tensor elements.
  /// Pointer to an array of size RANK(Weight). Each element specifies the size along the
  /// corresponding dimension
  uint32_t* blockSize;

  /// Array of size numBlocks of scale offset pairs.
  Qnn_ScaleOffset_t* scaleOffset;
} Qnn_BlockEncoding_t;

// clang-format off
/// Qnn_BlockEncoding_t initializer macro
#define QNN_BLOCK_ENCODING_INIT     \
  {                                 \
    0u,      /*blockSize*/          \
    NULL     /*scaleOffset*/        \
  }                                 \
// clang-format on

/**
 * @brief An enum to specify blockwise expansion block scale storage widths
 *
 */
typedef enum {
    QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8 = 0,
    QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16 = 1,
    // Unused, present to ensure 32 bits.
    QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_UNDEFINED = 0x7FFFFFFF
} Qnn_BlockwiseExpansionBlockScaleStorageType_t;

/**
 * @brief A struct to express block-wise expansion quantization parameters.
 *
 * @note This quantization encoding must not be used with dynamically shaped tensors.
 *
 */
typedef struct {
    /// The dimension (typically the channel dimension)
    int32_t axis;
    /// Array of size axisSize of scale offset pairs.
    Qnn_ScaleOffset_t* scaleOffsets;
    /// Number of blocks within the axis.
    uint32_t numBlocksPerAxis;
    /// Block bitwidth (e.g. 12 bits for 4 to 16 expansion)
    uint32_t blockScaleBitwidth;
    /// Size of the block scaling storage, must be able to store at least blockScaleBitwidth sized values.
    Qnn_BlockwiseExpansionBlockScaleStorageType_t blockScaleStorageType;
    union UNNAMED {
        /// A contiguous array of block scalings of size axisSize*numBlocksPerAxis. The array is laid out such that an element can be accessed via blocksScale8[axisIter*numBlocksPerAxis+blockIter].
        /// Used when blockStorageSize is QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8.
        uint8_t* blocksScale8;
        /// A contiguous array of block scalings of size axisSize*numBlocksPerAxis. The array is laid out such that an element can be accessed via blocksScale16[axisIter*numBlocksPerAxis+blockIter].
        /// Used when blockStorageSize is QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_16.
        uint16_t* blocksScale16;
    };
} Qnn_BlockwiseExpansion_t;

// clang-format off
/// Qnn_BlockScaleOffset_t initializer macro
#define QNN_BLOCKWISE_EXPANSION_INIT                                              \
  {                                                                               \
    0,                                                  /*axis*/                  \
    NULL,                                               /*scaleOffsets*/          \
    0u,                                                 /*numBlocksPerAxis*/      \
    0u,                                                 /*blockScaleBitwidth*/    \
    QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_UNDEFINED, /*blockScaleStorageType*/ \
    {                                                                             \
      NULL,                                             /*blocksScale8*/          \
    }                                                                             \
  }                                                                               \
// clang-format on

/**
 * @brief A struct to express vector quantization parameters.
 *
 * @note This quantization encoding is a specific case of per-channel quantization where
 * the weights and parameters are crafted in such a way to allow for compression and
 * codebook generation. For each group of rowsPerBlock*columnsPerBlock weights, there
 * will be 2^indexBitwidth unique vectorDimension-tuples of weights.
 *
 * @note This quantization encoding must not be used with dynamically shaped tensors.
 *
 */
typedef struct {
    /// Vector Quantization can be thought of as per-channel quantization with specifically
    /// crafted weights and encoding parameters that allow for codebook generation
    /// Each weight within the codebook is bwAxisScaleOffset.bitwidth bits wide
    Qnn_BwAxisScaleOffset_t bwAxisScaleOffset;
    /// Number of rows in the block of decoded weight coordinates
    uint32_t rowsPerBlock;
    /// Number of colums inf the block of decoded weight coordinates
    uint32_t columnsPerBlock;
    /// The dimension of the vector encoding. e.g 1D,2D,3D... for 1, 2 or 3 weights per index, respectively
    uint8_t vectorDimension;
    /// A value describing how the weights from a given lookup will be unpacked
    uint8_t vectorStride;
    /// The bitwidth of the each index into the codebook
    uint8_t indexBitwidth;
} Qnn_VectorEncoding_t;

// clang-format off
/// Qnn_VectorEncoding_t initializer macro
#define QNN_VECTOR_ENCODING_INIT                                                  \
  {                                                                               \
    QNN_BW_AXIS_SCALE_OFFSET_INIT,                        /*bwAxisScaleOffset*/   \
    0u,                                                   /*rowsPerBlock*/        \
    0u,                                                   /*columnsPerBlock*/     \
    0u,                                                   /*vectorDimension*/     \
    0u,                                                   /*vectorStride*/        \
    0u,                                                   /*indexBitwidth*/       \
  }                                                                               \
// clang-format on

/**
 * @brief A struct which defines the quantization parameters, and union of supported quantization
 * encoding structs.
 */
typedef struct {
  Qnn_Definition_t encodingDefinition;
  /// Quantization encoding type identifying quantization encoding structure to use
  Qnn_QuantizationEncoding_t quantizationEncoding;
  union UNNAMED {
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_SCALE_OFFSET. Note that this field is a value.
    Qnn_ScaleOffset_t scaleOffsetEncoding;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET. Note that this field is a value.
    Qnn_AxisScaleOffset_t axisScaleOffsetEncoding;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET. Note that this field is a value.
    Qnn_BwScaleOffset_t bwScaleOffsetEncoding;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET. Note that this field is a value.
    Qnn_BwAxisScaleOffset_t bwAxisScaleOffsetEncoding;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_BLOCK. Note that this field is a value.
    Qnn_BlockEncoding_t blockEncoding;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION. Note that this field is a pointer.
    Qnn_BlockwiseExpansion_t* blockwiseExpansion;
    /// Used when quantizationEncoding is QNN_QUANTIZATION_ENCODING_VECTOR. Note that this field is a pointer.
    Qnn_VectorEncoding_t* vectorEncoding;
  };
} Qnn_QuantizeParams_t;

// clang-format off
/// Qnn_QuantizeParams_t initializer macro
#define QNN_QUANTIZE_PARAMS_INIT                                      \
  {                                                                   \
    QNN_DEFINITION_UNDEFINED,                /*encodingDefinition*/   \
    QNN_QUANTIZATION_ENCODING_UNDEFINED,     /*quantizationEncoding*/ \
    {                                                                 \
      QNN_SCALE_OFFSET_INIT /*scaleOffsetEncoding*/                   \
    }                                                                 \
  }
// clang-format on

/**
 * @brief An n-dimensional tensor formatted in memory as flat buffer where the last dimension varies
 *        the fastest. Also known as a dense tensor.
 */
#define QNN_TENSOR_DATA_FORMAT_DENSE       0
#define QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER QNN_TENSOR_DATA_FORMAT_DENSE

/**
 * @brief An n-dimensional tensor formatted in memory as a sparse tensor. Sparse tensors may only be
 *        QNN_TENSOR_TYPE_NATIVE. Sparse tensors must also fully specify Qnn_SparseParams_t.
 */
#define QNN_TENSOR_DATA_FORMAT_SPARSE 1

// TODO: advertise the layout
/**
 * @brief A tensor formatted as a codebook. This tensor data format is to be used only in
 * conjunction with a quantized QNN_TENSOR_TYPE_STATIC tensor using the Qnn_VectorEncoding_t
 * encoding.
 */
#define QNN_TENSOR_DATA_FORMAT_CODEBOOK 2

/**
 * @brief Tensor data formatted in microscaling (MX) format. Compatible with multiple data types.
 */
#define QNN_TENSOR_DATA_FORMAT_MX 3

/**
* @brief An tensor compressed in memory in UBWC_RGBA8888 format, using the universal
 *       bandwidth compression (UBWC) scheme.
*/
#define QNN_TENSOR_DATA_FORMAT_UBWC_RGBA8888 4

/**
* @brief An tensor compressed in memory in UBWC_NV12 format, using the universal
 *       bandwidth compression (UBWC) scheme.
*/
#define QNN_TENSOR_DATA_FORMAT_UBWC_NV12 5

/**
* @brief An tensor compressed in memory in UBWC_NV12 format, using the universal
 *       bandwidth compression (UBWC) scheme. This data format particularly represents
 *       the Y plane of the NV12 format
*/
#define QNN_TENSOR_DATA_FORMAT_UBWC_NV12_Y 6

/**
* @brief An tensor compressed in memory in UBWC_NV12 format, using the universal
 *       bandwidth compression (UBWC) scheme. This data format particularly represents
 *       the UV plane of the NV12 format
*/
#define QNN_TENSOR_DATA_FORMAT_UBWC_NV12_UV 7

/**
 * @brief Tensor data formatted in native HMX weight format. This data format is desgined
 *        specifically for HMX weights.
 *        This format only supports the following datatype for now:
 *        UFIXED_UINT_8 with offset=128.
*/
#define QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT 8

/**
 * @brief Tensor data format identifier. The default format
 *        QNN_TENSOR_DATA_FORMAT_DENSE is supported by all backends. Backends may also support
 *        QNN_TENSOR_DATA_FORMAT_SPARSE or QNN_TENSOR_DATA_FORMAT_CODEBOOK.
 * @note  Data format for intermediate tensors, i.e ones of type QNN_TENSOR_TYPE_NATIVE
 *        may not be honored by a backend, because it can choose to pick a data format that is
 *        more conducive for its execution.
 */
typedef uint32_t Qnn_TensorDataFormat_t;

/**
 * @brief An enum specifying memory types of tensor data.
 */
typedef enum {
  /// Raw memory pointer
  QNN_TENSORMEMTYPE_RAW = 0,
  /// Memory object, provide capability for memory sharing in between QNN accelerator backends.
  QNN_TENSORMEMTYPE_MEMHANDLE = 1,
  /// Callback to retrieve a raw memory pointer
  QNN_TENSORMEMTYPE_RETRIEVE_RAW = 2,
  // Unused, present to ensure 32 bits.
  QNN_TENSORMEMTYPE_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorMemType_t;

/**
 * @brief A struct which defines a memory buffer
 *
 */
typedef struct {
  /// app-accessible data pointer, provided by app.
  void* data;
  /// size of buffer, in bytes, pointed to by data.
  uint32_t dataSize;
} Qnn_ClientBuffer_t;

/**
 * @brief A client-defined function used to obtain tensor data when the tensor memory type is
 *        QNN_TENSORMEMTYPE_RETRIEVE_RAW. Qnn_GetTensorRawDataRn_t may be called multiple times for
 * the same tensor. Each call to Qnn_GetTensorRawDataRn_t must be accompanied by a call to
 * Qnn_FreeTensorRawDataFn_t to free any allocated data for that tensor. It is not required that
 * this function be thread safe, unless needed to support retrieval of tensor resources that may be
 * shared between threads.
 *
 * @param[in] id The tensor ID.
 * @param[in] context the context to which the tensor is associated
 * @param[in] graph the graph to which the context is associated. For context tensors this field
 *            should be null.
 *
 * @param[out] clientBuf Pointer to the tensor's client buffer.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Client Buffer data successfully provided.
 *         - QNN_TENSOR_ERROR_DOES_NOT_EXIST: Tensor with __id__ does not exist or was not created
 * as QNN_TENSORMEMTYPE_RETRIEVE_RAW.
 *         - QNN_COMMON_ERROR_INVALID_ARGUMENT: __clientBuf__ is NULL
 *         - QNN_COMMON_ERROR_RESOURCE_UNAVAILABLE: Requested tensor data cannot be allocated.
 *
 */
typedef Qnn_ErrorHandle_t (*Qnn_GetTensorRawDataFn_t)(Qnn_ContextHandle_t context,
                                                      Qnn_GraphHandle_t graph,
                                                      uint64_t id,
                                                      Qnn_ClientBuffer_t* clientBuf);

/**
 * @brief A client-defined function used to free tensor data previously obtained by
 * Qnn_GetTensorDataFn_t. After the call to Qnn_FreeTensorDataFn_t the data provided in the client
 * buffer clientBuf should be considered invalid. If Qnn_GetTensorRawDataRn_t has been called
 * multiple times for the same tensor then Qnn_FreeTensorRawDataFn_t must be called an equivalent
 * number of times to free all allocated data for this tensor. It is not required that this function
 * be thread safe, unless needed to support releasing of tensor resources that may be shared between
 * threads.
 *
 * @param[in] id The tensor ID.
 * @param[in] context the context to which the tensor is associated
 * @param[in] graph the graph to which the context is associated. For context tensors this field
 * should be null.
 *
 * @return Error code:
 *         - QNN_SUCCESS: Client Buffer data successfully freed.
 *         - QNN_TENSOR_ERROR_DOES_NOT_EXIST: Tensor with __id__ does not exist, was not created
 *           as QNN_TENSORMEMTYPE_RETRIEVE_RAW, or has already been free'd.
 *
 */
typedef Qnn_ErrorHandle_t (*Qnn_FreeTensorRawDataFn_t)(Qnn_ContextHandle_t context,
                                                       Qnn_GraphHandle_t graph,
                                                       uint64_t id);

typedef struct {
  Qnn_GetTensorRawDataFn_t getTensorData;
  Qnn_FreeTensorRawDataFn_t freeTensorData;
} Qnn_TensorRetrieveRaw_t;

// clang-format off
/// Qnn_TensorDataRetrieve_t initializer macro
#define QNN_TENSOR_RETRIEVE_RAW_INIT \
  {                                  \
    NULL,   /*getTensorData*/        \
    NULL    /*freeTensorData*/       \
  }
// clang-format on

// clang-format off
/// Qnn_ClientBuffer_t initializer macro
#define QNN_CLIENT_BUFFER_INIT \
  {                            \
    NULL, /*data*/             \
    0u    /*dataSize*/         \
  }
// clang-format on

/**
 * @brief A struct which defines an opaque object
 *
 */
typedef struct {
  /// Data pointer to the opaque object
  void* data;
  /// Size of buffer, in bytes, pointed to by data
  uint64_t len;
} Qnn_OpaqueObject_t;

// clang-format off
/// Qnn_OpaqueObject_t initializer macro
#define QNN_OPAQUE_OBJECT_INIT \
  {                            \
    NULL, /*data*/             \
    0u    /*len*/              \
  }
// clang-format on

/**
 * @brief A struct which describes the properties of a V1 version of tensor.
 *
 */
typedef struct {
  /// Integer identifier for a tensor.
  uint32_t id;
  /// Tensor name.
  const char* name;
  /// Tensor type.
  Qnn_TensorType_t type;
  /// Tensor data formatting in memory (refer to definition type for info).
  Qnn_TensorDataFormat_t dataFormat;
  /// Tensor data type.
  Qnn_DataType_t dataType;
  /// Tensor quantization params.
  Qnn_QuantizeParams_t quantizeParams;
  /// Tensor rank.
  uint32_t rank;
  /// Tensor dimension array of length _rank_. For detailed behavior of dimensions field with
  /// various APIs, refer SDK documentation. Must be NULL when rank is 0.
  uint32_t* dimensions;
  /// Tensor memory type.
  Qnn_TensorMemType_t memType;
  /// Actual data contained in the tensor.
  union UNNAMED {
    /// Tensor data provided by client as a pointer to raw memory (see QNN_TENSORMEMTYPE_RAW).
    Qnn_ClientBuffer_t clientBuf;
    /// Tensor data shared via a memory handle (see QNN_TENSORMEMTYPE_MEMHANDLE).
    Qnn_MemHandle_t memHandle;
  };
} Qnn_TensorV1_t;

// clang-format off
/// Qnn_TensorV1_t initializer macro
#define QNN_TENSOR_V1_INIT                                        \
  {                                                               \
    0u,                                     /*id*/                \
    NULL,                                   /*name*/              \
    QNN_TENSOR_TYPE_UNDEFINED,              /*type*/              \
    QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,     /*dataFormat*/        \
    QNN_DATATYPE_UNDEFINED,                 /*dataType*/          \
    QNN_QUANTIZE_PARAMS_INIT,               /*quantizeParams*/    \
    0u,                                     /*rank*/              \
    NULL,                                   /*dimensions*/        \
    QNN_TENSORMEMTYPE_UNDEFINED,            /*memType*/           \
    {                                                             \
      QNN_CLIENT_BUFFER_INIT                /*clientBuf*/         \
    }                                                             \
  }
// clang-format on

/**
 * @brief An enum specifying sparse layout of a tensor. Used only when *dataFormat* is set to
 * QNN_TENSOR_DATA_FORMAT_SPARSE.
 */
typedef enum {
  /// Hybrid coordinate list sparse tensor layout
  QNN_SPARSE_LAYOUT_HYBRID_COO = 0,
  // Unused, present to ensure 32 bits.
  QNN_SPARSE_LAYOUT_UNDEFINED = 0x7FFFFFFF
} Qnn_SparseLayoutType_t;

/**
 * @brief A struct which defines the parameters for a COO sparse tensor layout.
 */
typedef struct {
  /// Number of specified elements of a sparse tensor. Treated as the maximum when creating a
  /// tensor.
  uint32_t numSpecifiedElements;
  /// Size of the index for a hybrid COO sparse tensor. The size of the index can range from 1 to
  /// the rank of the tensor. This feature allows for partially sparse tensors.
  uint32_t numSparseDimensions;
} Qnn_SparseLayoutHybridCoo_t;

// clang-format off
/// Qnn_SparseLayoutCoo_t initializer macro
#define QNN_SPARSE_LAYOUT_HYBRID_COO_INIT \
  {                                       \
    0u, /*numSpecifiedElements*/          \
    0u  /*numSparseDimensions*/           \
  }
// clang-format on

/**
 * @brief A struct which defines the sparse tensor parameters. See the SDK documentation for
 *        details. Used only when *dataFormat* is set to QNN_TENSOR_DATA_FORMAT_SPARSE.
 */
typedef struct {
  /// Specifies the sparse tensor layout
  Qnn_SparseLayoutType_t type;
  union UNNAMED {
    /// Hybrid coordinate list layout. Used when *type* is QNN_SPARSE_LAYOUT_HYBRID_COO.
    Qnn_SparseLayoutHybridCoo_t hybridCoo;
  };
} Qnn_SparseParams_t;

// clang-format off
/// Qnn_SparseParams_t initializer macro
#define QNN_SPARSE_PARAMS_INIT                      \
  {                                                 \
    QNN_SPARSE_LAYOUT_UNDEFINED,      /*type*/      \
    QNN_SPARSE_LAYOUT_HYBRID_COO_INIT /*hybridCoo*/ \
  }
// clang-format on

/**
 * @brief A struct which describes the properties of a V2 version of tensor.
 *
 */
typedef struct {
  /// Unique integer identifier for a tensor, generated by the backend based on the tensor name.
  uint32_t id;
  /// Unique tensor name.
  const char* name;
  /// Tensor type.
  Qnn_TensorType_t type;
  /// Tensor data formatting in memory (refer to definition type for info).
  Qnn_TensorDataFormat_t dataFormat;
  /// Tensor data type.
  Qnn_DataType_t dataType;
  /// Tensor quantization params.
  Qnn_QuantizeParams_t quantizeParams;
  /// Tensor rank. Note that rank cannot be dynamic.
  uint32_t rank;
  /// Tensor dimension array of length _rank_. For detailed behavior of dimensions field with
  /// various APIs, refer to their API documentation. Must be NULL when rank is 0. Must contain
  /// non-zero values if non-null.
  uint32_t* dimensions;
  /// Tensor memory type.
  Qnn_TensorMemType_t memType;
  /// Actual data contained in the tensor.
  union UNNAMED {
    /// Tensor data provided by client as a pointer to raw memory (see QNN_TENSORMEMTYPE_RAW).
    Qnn_ClientBuffer_t clientBuf;
    /// Tensor data shared via a memory handle (see QNN_TENSORMEMTYPE_MEMHANDLE).
    Qnn_MemHandle_t memHandle;
    /// Tensor data provided by client as a raw pointer retrieved through a callback
    /// (QNN_TENSORMEMTYPE_RETRIEVE_RAW)
    Qnn_TensorRetrieveRaw_t* retrieveRaw;
  };
  /// A boolean array of length _rank_ indicating if a tensor dimension is dynamic. Must be NULL
  /// when rank is 0. Can be NULL if all dimensions are static. A true (non-zero) value indicates
  /// the corresponding dimension is dynamic and a false (zero) value indicates the corresponding
  /// dimension is static. Note that QNN_TENSOR_TYPE_STATIC tensors (see _type_) cannot have dynamic
  /// dimensions. Support for this field can be queried via
  /// QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS. If this field is unsupported, it must be NULL.
  uint8_t* isDynamicDimensions;
  /// Sparse tensor parameters. Pertains only to sparse tensors (see QNN_TENSOR_DATA_FORMAT_SPARSE).
  /// Support for this field can be queried via QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY.
  Qnn_SparseParams_t sparseParams;
  /// Indicates whether or not a call to QnnGraph_execute[Async] produced this output tensor.
  /// Applicable only to QNN_TENSOR_TYPE_APP_READ and QNN_TENSOR_TYPE_APP_READWRITE tensor types.
  /// This field will be undefined if QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION is not
  /// supported. Otherwise, this field is not used.
  uint8_t isProduced;
} Qnn_TensorV2_t;

// clang-format off
/// Qnn_TensorV2_t initializer macro
#define QNN_TENSOR_V2_INIT                                     \
  {                                                            \
    0u,                                 /*id*/                 \
    NULL,                               /*name*/               \
    QNN_TENSOR_TYPE_UNDEFINED,          /*type*/               \
    QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/         \
    QNN_DATATYPE_UNDEFINED,             /*dataType*/           \
    QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/     \
    0u,                                 /*rank*/               \
    NULL,                               /*dimensions*/         \
    QNN_TENSORMEMTYPE_UNDEFINED,        /*memType*/            \
    {                                                          \
      QNN_CLIENT_BUFFER_INIT            /*clientBuf*/          \
    },                                                         \
    NULL,                               /*isDynamicDimension*/ \
    QNN_SPARSE_PARAMS_INIT,             /*sparseParams*/       \
    0u                                  /*isProduced*/         \
  }
// clang-format on

/**
 * @brief Enum to distinguish various tensor versions
 */
typedef enum {
  /// Enum to choose usage of Qnn_TensorV1_t in Qnn_Tensor_t
  QNN_TENSOR_VERSION_1 = 1,
  /// Enum to choose usage of Qnn_TensorV2_t in Qnn_Tensor_t
  QNN_TENSOR_VERSION_2 = 2,
  // Unused, present to ensure 32 bits.
  QNN_TENSOR_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorVersion_t;

/**
 * @brief A struct which provides various versions of a tensor
 */
typedef struct {
  /// Version of the QNN tensor
  Qnn_TensorVersion_t version;
  union UNNAMED {
    /// Tensor version 1 (see QNN_TENSOR_VERSION_1)
    Qnn_TensorV1_t v1;
    /// Tensor version 2 (see QNN_TENSOR_VERSION_2)
    Qnn_TensorV2_t v2;
  };
} Qnn_Tensor_t;

/// Qnn_Tensor_t initializer macro
#define QNN_TENSOR_INIT               \
  {                                   \
    QNN_TENSOR_VERSION_1, /*version*/ \
    {                                 \
      QNN_TENSOR_V1_INIT /*v1*/       \
    }                                 \
  }

/**
 * @brief A struct which describes the properties of a V1 set of input and output tensors
 *
 */
typedef struct {
  /// The number of input tensors.
  uint32_t numInputs;
  /// Array of input tensors.
  Qnn_Tensor_t* inputs;
  /// The number of output tensors.
  uint32_t numOutputs;
  /// Array of output tensors.
  Qnn_Tensor_t* outputs;
} Qnn_TensorSetV1_t;

// clang-format off
/// Qnn_TensorSetV1_t initializer macro
#define QNN_TENSOR_SET_V1_INIT \
  {                            \
    0u,   /*inputs*/           \
    NULL, /*inputTensors*/     \
    0u,   /*numOutputs*/       \
    NULL  /*outputs*/          \
  }
// clang-format on

/**
 * @brief Enum to distinguish between tensor set versions
 */
typedef enum {
  /// Enum to choose usage of Qnn_TensorSetV1_t in Qnn_TensorSet_t
  QNN_TENSOR_SET_VERSION_1 = 1,
  /// Unused, present to ensure 32 bits.
  QNN_TENSOR_SET_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_TensorSetVersion_t;

/**
 * @brief A struct which provides the version of a tensor set
 */
typedef struct {
  /// Version of the QNN tensor set
  Qnn_TensorSetVersion_t version;
  union UNNAMED {
    /// Tensor set version 1 (see QNN_TENSOR_SET_VERSION_1)
    Qnn_TensorSetV1_t v1;
  };
} Qnn_TensorSet_t;

/// Qnn_TensorSet_t initializer macro
#define QNN_TENSOR_SET_INIT               \
  {                                       \
    QNN_TENSOR_SET_VERSION_1, /*version*/ \
    {                                     \
      QNN_TENSOR_SET_V1_INIT /*v1*/       \
    }                                     \
  }

/**
 * @brief A struct which defines a named scalar or tensor parameter.
 *
 */
typedef struct {
  /// Parameter type: scalar or tensor
  Qnn_ParamType_t paramType;
  /// Name of the parameter
  const char* name;

  union UNNAMED {
    /// Scalar parameter specification
    Qnn_Scalar_t scalarParam;
    /// Tensor parameter specification; tensors referred to must be STATIC.
    Qnn_Tensor_t tensorParam;
  };
} Qnn_Param_t;

// clang-format off
/// Qnn_Param_t initializer macro
#define QNN_PARAM_INIT                     \
  {                                        \
    QNN_PARAMTYPE_UNDEFINED, /*paramType*/ \
    NULL,                    /*name*/      \
    {                                      \
      QNN_SCALAR_INIT /*scalarParam*/      \
    }                                      \
  }
// clang-format on

/**
 * @brief This struct defines the configuration for a single operation.
 */
typedef struct {
  /// A human-readable name for the operation instance.
  const char* name;
  /// The name of the operation package to which this operation's type belongs.
  const char* packageName;
  /// The name of operation type (e.g. Conv2D).
  const char* typeName;
  /// The number of static parameters provided in the params array.
  uint32_t numOfParams;
  /// Array of operation parameters.
  Qnn_Param_t* params;
  /// The number of input tensors.
  uint32_t numOfInputs;
  /// Array of input tensors.
  Qnn_Tensor_t* inputTensors;
  /// The number of output tensors.
  uint32_t numOfOutputs;
  /// Array of output tensors.
  Qnn_Tensor_t* outputTensors;
} Qnn_OpConfigV1_t;

// clang-format off
/// Qnn_OpConfigV1_t initializer macro
#define QNN_OPCONFIG_V1_INIT    \
  {                             \
    NULL,     /*name*/          \
    NULL,     /*packageName*/   \
    NULL,     /*typeName*/      \
    0u,       /*numOfParams*/   \
    NULL,     /*params*/        \
    0u,       /*numOfInputs*/   \
    NULL,     /*inputTensors*/  \
    0u,       /*numOfOutputs*/  \
    NULL      /*outputTensors*/ \
  }
// clang-format on

/**
 * @brief Enum to distinguish various opConfig versions
 */
typedef enum {
  /// Enum to choose usage of Qnn_OpConfigV1_t in Qnn_OpConfig_t
  QNN_OPCONFIG_VERSION_1 = 1,
  // Unused, present to ensure 32 bits.
  QNN_OPCONFIG_VERSION_UNDEFINED = 0x7FFFFFFF
} Qnn_OpConfigVersion_t;

/**
 * @brief Structure which provides various versions of an opConfig
 */
typedef struct {
  /// Version of the QNN opConfig
  Qnn_OpConfigVersion_t version;
  union UNNAMED {
    /// Op config version 1 (see QNN_OPCONFIG_VERSION_1)
    Qnn_OpConfigV1_t v1;
  };
} Qnn_OpConfig_t;

// clang-format off
/// Qnn_OpConfig_t initializer macro
#define QNN_OPCONFIG_INIT               \
  {                                     \
    QNN_OPCONFIG_VERSION_1, /*version*/ \
    {                                   \
      QNN_OPCONFIG_V1_INIT /*v1*/       \
    }                                   \
  }
// clang-format on

/**
 * @brief An enum which identifies SOC models.
 *
 * @deprecated This enumeration will no longer be updated.
 */
typedef enum {
  QNN_SOC_MODEL_UNKNOWN = 0,

  QNN_SOC_MODEL_SDM845  = 1,
  QNN_SOC_MODEL_SDM835  = 2,
  QNN_SOC_MODEL_SDM821  = 3,
  QNN_SOC_MODEL_SDM820  = 4,
  QNN_SOC_MODEL_SDM801  = 5,
  QNN_SOC_MODEL_SDM670  = 6,
  QNN_SOC_MODEL_SDM660  = 7,
  QNN_SOC_MODEL_SDM652  = 8,
  QNN_SOC_MODEL_SDM636  = 9,
  QNN_SOC_MODEL_SDM630  = 10,
  QNN_SOC_MODEL_SDM625  = 11,
  QNN_SOC_MODEL_SDM855  = 12,
  QNN_SOC_MODEL_SDM710  = 13,
  QNN_SOC_MODEL_SDM632  = 15,
  QNN_SOC_MODEL_SM6150  = 16,
  QNN_SOC_MODEL_SM7150  = 17,
  QNN_SOC_MODEL_QCS405  = 18,
  QNN_SOC_MODEL_SM6125  = 19,
  QNN_SOC_MODEL_QCS403  = 20,
  QNN_SOC_MODEL_SDM865  = 21,
  QNN_SOC_MODEL_IPQ6018 = 23,
  QNN_SOC_MODEL_IPQ6028 = 24,
  QNN_SOC_MODEL_SM7250  = 25,
  QNN_SOC_MODEL_SA8195  = 26,
  QNN_SOC_MODEL_SM6250  = 27,
  QNN_SOC_MODEL_SM4250  = 28,
  QNN_SOC_MODEL_SM6350  = 29,
  QNN_SOC_MODEL_SM8350  = 30,
  QNN_SOC_MODEL_SM4350  = 31,
  QNN_SOC_MODEL_SM7350  = 32,
  QNN_SOC_MODEL_QCS410  = 33,
  QNN_SOC_MODEL_SM8325  = 34,
  QNN_SOC_MODEL_SM7325  = 35,
  QNN_SOC_MODEL_SM8450  = 36,
  QNN_SOC_MODEL_SC8280X = 37,
  QNN_SOC_MODEL_SM7315  = 38,
  QNN_SOC_MODEL_SA8295  = 39,
  QNN_SOC_MODEL_SM6225  = 40,
  QNN_SOC_MODEL_SM7450  = 41,
  QNN_SOC_MODEL_SM8475  = 42,
  QNN_SOC_MODEL_SM8550  = 43,
  QNN_SOC_MODEL_SXR1230P = 45,
  QNN_SOC_MODEL_SSG2115P = 46,
  QNN_SOC_MODEL_STP6225P = 47,
  QNN_SOC_MODEL_QCS6125  = 48,
  QNN_SOC_MODEL_QRB4210  = 49,
  QNN_SOC_MODEL_SM6450   = 50,
  QNN_SOC_MODEL_QCS7230  = 51,
  QNN_SOC_MODEL_SA8255   = 52,
  QNN_SOC_MODEL_SXR2230P = 53,
  QNN_SOC_MODEL_SM7475   = 54,
  QNN_SOC_MODEL_SM4375   = 55,
  QNN_SOC_MODEL_QCM4325  = 56,
  QNN_SOC_MODEL_SM8650   = 57,
  QNN_SOC_MODEL_SSG2125P = 58,
  QNN_SOC_MODEL_SM4450   = 59,
  QNN_SOC_MODEL_SC8380XP = 60,
  QNN_SOC_MODEL_SM7435   = 61,
  QNN_SOC_MODEL_SA8540   = 62,
  QNN_SOC_MODEL_AIC100   = 63,
  QNN_SOC_MODEL_SM7550   = 64,
  QNN_SOC_MODEL_SM6450Q  = 65,
  QNN_SOC_MODEL_QCS8550  = 66,
  QNN_SOC_MODEL_SA8620P  = 67,
  QNN_SOC_MODEL_SM8635   = 68,
  QNN_SOC_MODEL_SM8750   = 69,
  QNN_SOC_MODEL_SM7675   = 70,
  QNN_SOC_MODEL_SM4635   = 71,
  QNN_SOC_MODEL_SA8797   = 72,
  QNN_SOC_MODEL_SM7635   = 73,
  QNN_SOC_MODEL_SM6650   = 74,
  QNN_SOC_MODEL_SXR2330P = 75,
  QNN_SOC_MODEL_SM6475   = 76,
  QNN_SOC_MODEL_QCS9100  = 77,
  QNN_SOC_MODEL_QCM6690  = 78,
  QNN_SOC_MODEL_IPQ9574  = 79,
  QNN_SOC_MODEL_IPQ5404  = 80,
  QNN_SOC_MODEL_IPQ5424  = 81,
  QNN_SOC_MODEL_QCS8300  = 82,
  QNN_SOC_MODEL_QCS2290  = 83,
  QNN_SOC_MODEL_SA525M   = 84,
  QNN_SOC_MODEL_SM8735   = 85,
  QNN_SOC_MODEL_SM7750   = 86,
  QNN_SOC_MODEL_SM8850   = 87,

  QNN_SOC_MODEL_DYNAMIC_SDM = INT_MAX

} Qnn_SocModel_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_TYPES_H
