//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 *  @file
 *  @brief QNN HTP component Graph API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnGraph.h for HTP backend
 */

#ifndef QNN_HTP_GRAPH_H
#define QNN_HTP_GRAPH_H

#include "QnnGraph.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
/**
 * @brief QnnHtpGraph config value macro. Represents to use the maximum
 *        available number of the resource.
 *
 *        Currently only applicable for QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE.
 */
#define QNN_HTP_GRAPH_CONFIG_OPTION_MAX 0

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief This enum provides different HTP graph optimization
 *        options that can be used to finalize the graph
 *        for optimum performance.
 */
typedef enum {
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD                = 1,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES                  = 2,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG        = 3,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                       = 4,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS               = 5,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SPARSE_WEIGHTS_COMPRESSION = 6,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SLC_ALLOCATOR              = 7,
  QNN_HTP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN                           = 0x7fffffff
} QnnHtpGraph_OptimizationType_t;

// clang-format off

/**
 * @brief Struct describing the set of optimization types
 *        and the values associated with each optimization type.
 *
 *        Below is the Map between QnnHtpGraph_OptimizationType_t and allowable values:
 *
 *        \verbatim embed:rst:leading-asterisk
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | #  | OptimizationType option                                            | Allowable values                                                    |
 *        +====+====================================================================+=====================================================================+
 *        | 1  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_SCHEDULE_THRESHOLD                 | Reserved                                                            |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 2  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_RETRIES                   | Reserved                                                            |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 3  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG         | Defines the optimization strategy used by the HTP backend           |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 4  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC                        | Reserved                                                            |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 5  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_DLBC_WEIGHTS                | Enables DLBC weights compression                                    |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 6  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SPARSE_WEIGHTS_COMPRESSION  | Enables Weight Sparsity Compression                                 |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        | 7  | QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SLC_ALLOCATOR               | Enables System Level Cache Allocator usage                          |
 *        +----+--------------------------------------------------------------------+---------------------------------------------------------------------+
 *        \endverbatim
 */
typedef struct {
  QnnHtpGraph_OptimizationType_t type;
  float floatValue;
} QnnHtpGraph_OptimizationOption_t;

/**
 * @brief This struct encapsulates all the VTCM configurations for parallel graph execution.
 *
 * @code
 *    |<--           (1) 8MB Total Hardware VTCM           -->|
 *           |<--            (2) 7MB Addressable           -->|
 *    +------+------+------+------+------+------+------+------+
 *    |  CV  |      |      |      |      |      |      |      |
 *    +------+------+------+------+------+------+------+------+
 *           |<-- (4) Graph A  -->|<--     (4) Graph B     -->|
 *
 *         A |> 0 MB      (3) Graph Offset
 *         B |-------------------> 3 MB
 * @endcode
 */
typedef struct {
    /// (4) above, the amount of VTCM used by a graph
    uint32_t sizeInBytes;
    /// (3) above, where in the addressable region to start VTCM.
    ///     Note: (3) + (4) <= (2)
    uint32_t offsetInBytes;
    /// (2) Addressable portion of VTCM.
    /// Set to less than hardware size so Graph(s) can coexist with other VTCM clients.
    uint32_t sizeTotalInBytes;

    // For ABI compatibility in the future.
    // Set to 0 for now.
    uint32_t reserved[3];
} QnnHtpGraph_VtcmConfig_t;

/**
 * @brief This enum defines whether graph concurrency (i.e. multiple graphs running concurrently)
 *        is possible, and how to behave when circumstances for concurrency aren't possible.
 */
typedef enum {
  /// This graph will not be able to run concurrently with other graphs.
  QNN_HTP_GRAPH_CONCURRENCY_OPTION_NONE                       = 0,
  QNN_HTP_GRAPH_CONCURRENCY_OPTION_DEFAULT                    = QNN_HTP_GRAPH_CONCURRENCY_OPTION_NONE,
  /// Graph will try to run concurrently, sharing all resources on the DSP (VTCM, HMX, HVX, etc).
  QNN_HTP_GRAPH_CONCURRENCY_OPTION_ALL_SHARED                 = 1,
  // Unused, present to ensure 32 bits.
  QNN_HTP_GRAPH_CONCURRENCY_OPTION_UNKNOWN                    = 0x7fffffff
} QnnHtpGraph_ConcurrencyOption_t;

/**
 * @brief This struct encapsulates all the configurations for parallel graph execution.
 */
typedef struct {
  QnnHtpGraph_ConcurrencyOption_t concurrency;
  QnnHtpGraph_VtcmConfig_t vtcmConfig;

  // For ABI compatibility in the future.
  // Set to 0 for now.
  uint32_t reserved[4];
} QnnHtpGraph_ParallelGraphExecutionConfig_t;
/// The settings in this struct is only applicable
///  for DSP architectures >= V81.
/// Use on other SOCs will return an error.
///
/// Values will be defaulted to their SOC's TURBO frequency
///  (SOC as identified by Qnn_DeviceHandle_t).
///
/// On automotive SDKs HMX OP Bounding will be enabled by default.
///
/// On non-automotive SDKs using this setting will enable
///  HMX OP Bounding. It is off by default.
typedef struct QnnHtp_HmxBoundingInfo {
  /// Target HMX freq in Hz.
  /// Can be derived from sysMonApp (HexagonSDK) or QProfiler.
  float targetHmxFreqHz;
  /// Target DSP Core freq in Hz.
  /// Can be derived from sysMonApp (HexagonSDK) or QProfiler.
  float targetDspCoreFreq;
} QnnHtp_HmxBoundingInfo_t;

/// QnnHtpGraph_OptimizationOption_t initializer macro
#define QNN_HTP_GRAPH_OPTIMIZATION_OPTION_INIT              \
  {                                                         \
    QNN_HTP_GRAPH_OPTIMIZATION_TYPE_UNKNOWN, /*type*/       \
    0.0f                                     /*floatValue*/ \
  }
// clang-format on

/**
 * @brief This enum provides different HTP graph configuration
 *        options associated with QnnGraph
 */
typedef enum {
  QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION    = 1,
  QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION       = 2,
  QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB = 3,
  QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE       = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB,
  QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF = 4,
  QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF        = 5,
  QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS                    = 6,
  QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG                    = 7,
  QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES                          = 8,
  QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG    = 9,
  QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_BYTES                 = 10,
  QNN_HTP_GRAPH_CONFIG_OPTION_HMX_BOUNDING                       = 11,
  QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING                    = 12,
  QNN_HTP_GRAPH_CONFIG_OPTION_ASSUME_SAME_QUANT                  = 13,
  QNN_HTP_GRAPH_CONFIG_OPTION_SHARE_IO_BUFFER                    = 14,
  QNN_HTP_GRAPH_CONFIG_OPTION_RESERVED                           = 0x7fff0000,
  QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN                            = 0x7fffffff
} QnnHtpGraph_ConfigOption_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

/**
 * @brief A struct for different config parameters in a key value format.
 */
typedef struct {
  const char* key;
  Qnn_Scalar_t value;
} QnnHtpGraph_FinalizeConfig_t;

/**
 * @brief        Structure describing the set of configurations supported by graph.
 *               Objects of this type are to be referenced through QnnGraph_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of corresponding config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpGraph_ConfigOption_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | #  | Config Option | Configuration Struct/value                     |
 *               +====+=====================================================================================+================================================+
 *               | 1  | QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION | QnnHtpGraph_OptimizationOption_t
 * |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 2  | QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION | Qnn_Precision_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 3  |
 * QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB/QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE   | uint32_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 4  | QNN_HTP_GRAPH_CONFIG_OPTION_FOLD_RELU_ACTIVATION_INTO_CONV_OFF | bool |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 5  | QNN_HTP_GRAPH_CONFIG_OPTION_SHORT_DEPTH_CONV_ON_HMX_OFF | bool |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 6  | QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS | uint32_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 7  | QNN_HTP_GRAPH_CONFIG_OPTION_FINALIZE_CONFIG | QnnHtpGraph_FinalizeConfig_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 8  | QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES | uint32_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               |  9 | QNN_HTP_GRAPH_CONFIG_OPTION_PARALLEL_GRAPH_EXECUTION_CONFIG |
 * QnnHtpGraph_ParallelGraphExecutionConfig_t     |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 10 | QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_BYTES | uint32_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 11 | QNN_HTP_GRAPH_CONFIG_OPTION_HMX_BOUNDING | uint32_t |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 12 | QNN_HTP_GRAPH_CONFIG_OPTION_WEIGHTS_PACKING | bool |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 13 | QNN_HTP_GRAPH_CONFIG_OPTION_ASSUME_SAME_QUANT | bool |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               | 14 | QNN_HTP_GRAPH_CONFIG_OPTION_SHARE_IO_BUFFER | bool |
 *               +----+-------------------------------------------------------------------------------------+------------------------------------------------+
 *               +-------------------------+----------------------------------------------------------------+------------------------------------------------+
 *               | 0x7fff0000 - 0x7ffffffe | QNN_HTP_GRAPH_CONFIG_OPTION_RESERVED | These are
 * reserved for internal purposes       |
 *               +-------------------------+----------------------------------------------------------------+------------------------------------------------+
 *               \endverbatim
 *
 *               NOTE: Option #6 (i.e. QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS), can only be
 *               set prior to the first execution of the graph. Proceeding executions will not use
 *               the updated value if user does change it after the first execution.
 */
typedef struct {
  QnnHtpGraph_ConfigOption_t option;
  union {
    QnnHtpGraph_OptimizationOption_t optimizationOption;
    Qnn_Precision_t precision;
    uint32_t vtcmSizeInMB;
    bool foldReluActivationIntoConvOff;
    bool shortDepthConvOnHmxOff;
    uint64_t numHvxThreads;
    void* reserved;
    QnnHtpGraph_FinalizeConfig_t finalizeConfig;
    uint32_t numCores;
    QnnHtpGraph_ParallelGraphExecutionConfig_t parallelGraphExecutionConfig;
    uint32_t vtcmSizeInBytes;
    QnnHtp_HmxBoundingInfo_t hmxBoundingInfo;
    bool weightsPacking;
    bool assumeSameQuant;
    bool shareIOBuffer;
  };
} QnnHtpGraph_CustomConfig_t;

// clang-format on
/// QnnHtpGraph_CustomConfig_t initializer macro
#define QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT                            \
  {                                                                 \
    QNN_HTP_GRAPH_CONFIG_OPTION_UNKNOWN, /*option*/                 \
    {                                                               \
      QNN_HTP_GRAPH_OPTIMIZATION_OPTION_INIT /*optimizationOption*/ \
    }                                                               \
  }

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
