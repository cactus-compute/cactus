//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.s
//
//==============================================================================

/**
 *  @file
 *  @brief QNN HTP component Context API.
 *
 *         The interfaces in this file work with the top level QNN
 *         API and supplements QnnContext.h for HTP backend
 */

#ifndef QNN_HTP_CONTEXT_H
#define QNN_HTP_CONTEXT_H

#include "QnnContext.h"

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
 * @brief This enum provides different HTP context configuration
 *        options associated with QnnContext
 */
typedef enum {
  QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED               = 1,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS              = 2,
  QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET              = 3,
  QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED         = 4,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES                      = 5,
  QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION                    = 6,
  QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY                         = 7,
  QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION                    = 8,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION    = 9,
  QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE    = 10,
  QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA                    = 11,
  QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_CONCURRENT_RESOURCE_SHARING = 12,
  QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN                              = 0x7fffffff
} QnnHtpContext_ConfigOption_t;

typedef struct {
  // Handle referring to the first context associated to a group. When a new
  // group is to be registered, the following value must be 0.
  Qnn_ContextHandle_t firstGroupHandle;
  // Max spill-fill buffer to be allocated for the group of context in bytes.
  // The value that is passed during the registration of the first context to
  // a group is taken. Subsequent configuration of this value is disregarded.
  uint64_t maxSpillFillBuffer;
} QnnHtpContext_GroupRegistration_t;

// This enum is supported only with the QnnContext_createFromBinaryListAsync API, when
// shareResources is true; otherwise, it is ignored. This enumeration allows users to specify how
// graphs are going to be executed, providing QNN with hints for optimizing memory.
typedef enum {
  // Default value if no user input is provided.
  // This type is used for sequential graph execution, optimizing both VA and memory.
  SEQUENTIAL_WITH_VA_OPTIMIZATION,
  // This type is used for sequential graph execution, optimizing memory.
  SEQUENTIAL_WITHOUT_VA_OPTIMIZATION,
  // This type is used for concurrent resource sharing, optimizing memory by sharing
  // resources across contexts with the same priority level.
  CONCURRENT_OPTIMIZATION,
} QnnHtpContext_ShareResourcesOptimizationType_t;

//=============================================================================
// Public Functions
//=============================================================================

//------------------------------------------------------------------------------
//   Implementation Definition
//------------------------------------------------------------------------------

// clang-format off

/**
 * @brief        Structure describing the set of configurations supported by context.
 *               Objects of this type are to be referenced through QnnContext_CustomConfig_t.
 *
 *               The struct has two fields - option and a union of config values
 *               Based on the option corresponding item in the union can be used to specify
 *               config.
 *
 *               Below is the Map between QnnHtpContext_CustomConfig_t and config value
 *
 *               \verbatim embed:rst:leading-asterisk
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | #  | Config Option                                                       | Configuration Struct/value                       |
 *               +====+=====================================================================+==================================================+
 *               | 1  | QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED                | bool                                             |
 *               +====+=====================================================================+==================================================+
 *               | 2  | QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS               | QnnHtpContext_GroupRegistration_t                |
 *               +====+=====================================================================+==================================================+
 *               | 3  | QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET               | uint64_t                                         |
 *               +====+=====================================================================+==================================================+
 *               | 4  | QNN_HTP_CONTEXT_CONFIG_OPTION_DSP_MEMORY_PROFILING_ENABLED          | bool                                             |
 *               +====+=====================================================================+==================================================+
 *               | 5  | QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES                       | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 6  | QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION                     | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 7  | QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY                          | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 8  | QNN_HTP_CONTEXT_CONFIG_OPTION_INIT_ACCELERATION                     | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 9  | QNN_HTP_CONTEXT_CONFIG_OPTION_SKIP_VALIDATION_ON_BINARY_SECTION     | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 10 | QNN_HTP_CONTEXT_CONFIG_OPTION_SHARE_RESOURCES_OPTIMIZATION_TYPE     | QnnHtpContext_ShareResourcesOptimizationType_t   |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 11 | QNN_HTP_CONTEXT_CONFIG_OPTION_USE_EXTENDED_UDMA                     | bool                                             |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               | 12 | QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_CONCURRENT_RESOURCE_SHARING  | QnnHtpContext_GroupRegistration_t                |
 *               +----+---------------------------------------------------------------------+--------------------------------------------------+
 *               \endverbatim
 */
typedef struct QnnHtpContext_CustomConfig {
  QnnHtpContext_ConfigOption_t option;
  union UNNAMED {
    // This field sets the weight sharing which is by default false
    bool weightSharingEnabled;
    QnnHtpContext_GroupRegistration_t groupRegistration;
    // - Init time may be impacted depending the value set below
    // - Value should be grather than 0 and less than or equal to the file size
    //    - If set to 0, the feature is not utilized
    //    - If set to greater than file size, min(fileSize, fileReadMemoryBudgetInMb) is used
    // - As an example, if value 2 is passed, it would translate to (2 * 1024 * 1024) bytes
    uint64_t fileReadMemoryBudgetInMb;
    bool dspMemoryProfilingEnabled;
    // This field enables resource optimization. When it is set to true optimizations are
    // done based on QnnHtpContext_ShareResourcesOptimizationType_t setting.
     // Note This configuration option is only supported when using QnnContext_createFromBinaryListAsync API.
    bool shareResources;
    // This field enables I/O memory estimation during QnnContext_createFromBinary API when multiple
    // PDs are available. When enabled, it estimates the total size of the I/O tensors required by
    // the context to ensure sufficient space on the PD before deserialization. This feature helps
    // with memory registration failures in large models.
    // Note that enabling this feature increases peak RAM usage during context initialization phase
    // in QnnContext_createFromBinary, but sustained RAM remains unaffected.
    bool ioMemEstimation;
    // This field enables model preparation without mapping its content on the DSP side. It is
    // useful when a model needs to be prepared on the device but executed through a serialized
    // binary method. This prevents extra mapping onto the DSP VA space. Set this flag only when
    // creating the context.
    bool isPrepareOnly;
    // This field enables initialization acceleration, which is disabled by default.
    // If set to true, the DSP will utilize all hardware threads to accelerate deserialization.
    // It is not recommended to execute graphs simultaneously, as this will significantly degrade
    // performance.
    // Note that this feature may not be effective for small graphs with a few number of ops.
    bool initAcceleration;
    // This field enables crc32 check skip in Lora super adapter apply, which is disabled by default.
    // If set to true, crc32 check for non-base adapter in super adapter apply use case will be
    // skipped to improve time cost.
    // Note that base adapter in super adaper never do crc32 check, therefore, their apply time cost
    // won't improve by turning this config option on.
    bool skipValidationOnBinarySection;
    // If shareResources is true:
    // shareResOptType is read. If no value is set by the user,
    // the default value of QnnHtpContext_ShareResourcesOptimizationType_t is used.
    // If shareResources is false:
    // shareResOptType is ignored.
    // Note: This configuration option is only supported when using the QnnContext_createFromBinaryListAsync API.
    QnnHtpContext_ShareResourcesOptimizationType_t shareResOptType;
    // This field enables preparing graphs, associated with this context, with far-mapping enabled so that weights
    // and spill/fill buffer are mapped to the far region of the DSP which is helpful if PD's limited VA space is
    // exhausted. Total RAM usage may increase if used together with shared weights. Only available for Hexagon
    // arch v81 and above.
    bool useExtendedUdma;
    // This field enables concurrent resource sharing among graphs with the same priority level
    // during the QnnContext_createFromBinary API on devices that support this capability.
    QnnHtpContext_GroupRegistration_t concurrentGroupRegistration;
  };
} QnnHtpContext_CustomConfig_t;

/// QnnHtpContext_CustomConfig_t initializer macro
#define QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT            \
  {                                                   \
    QNN_HTP_CONTEXT_CONFIG_OPTION_UNKNOWN, /*option*/ \
    {                                                 \
      false                          /*weightsharing*/\
    }                                                 \
  }

/**
 * @brief Structure describing the set of properties supported by context.
 *        Objects of this type are to be referenced through QnnContext_CustomProperty_t.
 *        Used by QnnContext_getProperty.
 */
typedef enum {
  // get the alignment requirement of persistent buffers
  QNN_HTP_CONTEXT_GET_PROP_BUFFER_START_ALIGNMENT = 1,
  // get the size requirement of spill/fill buffer
  QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE = 2,
  // get the size requirement of persistent weights buffer
  QNN_HTP_CONTEXT_GET_PROP_WEIGHTS_BUFFER_SIZE = 3,
  // Unused, present to ensure 32 bits.
  QNN_HTP_CONTEXT_GET_PROP_UNDEFINED = 0x7fffffff
} QnnHtpContext_GetPropertyOption_t;

// used by QnnContext_getProperty
typedef struct {
  QnnHtpContext_GetPropertyOption_t option;
  union UNNAMED {
    uint64_t bufferStartAlignment;
    uint64_t spillfillBufferSize;
    uint64_t weightsBufferSize;
  };
} QnnHtpContext_CustomProperty_t;

// clang-format off
/// QnnHtpContext_CustomProperty_t initializer macro
#define QNN_HTP_CONTEXT_CUSTOM_PROPERTY_INIT                         \
  {                                                                  \
    QNN_HTP_CONTEXT_GET_PROP_UNDEFINED,                 /*option*/   \
    0                                         /*scratchBufferSize*/  \
  }
// clang-format on

// clang-format on
#ifdef __cplusplus
}  // extern "C"
#endif

#endif
