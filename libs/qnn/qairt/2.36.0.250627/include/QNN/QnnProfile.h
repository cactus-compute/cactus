//==============================================================================
//
// Copyright (c) 2019-2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief Profile component API.
 *
 *          Requires Backend to be initialized.
 *          Provides means to profile QNN backends to evaluate performance
 *          (memory and timing) of graphs and operations.
 */

#ifndef QNN_PROFILE_H
#define QNN_PROFILE_H

#include "QnnCommon.h"
#include "QnnTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Macros
//=============================================================================
/**
 * @brief QnnProfile_EventType_t definition to get stats related to creation of
 *        context and graphs. If supported, this profile data captures stats
 *        starting with the context creation (QnnContext_create) and ending with
 *        graph finalize (QnnGraph_finalize). Alternatively, in case of loading
 *        a cached context, it captures stats for creating context from the
 *        cache (QnnContext_createFromBinary).
 *
 * @note init information maybe available on both QNN_PROFILE_LEVEL_BASIC and
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note If unit information is not available, the value should be interpreted
 *       as time in microseconds.
 */
#define QNN_PROFILE_EVENTTYPE_INIT 100

/**
 * @brief QnnProfile_EventType_t definition to get stats related to finalize
 *        operation on graphs in a context. If supported, this profile data
 *        captures stats for graph finalize (QnnGraph_finalize).
 *
 * @note finalize information maybe available on both QNN_PROFILE_LEVEL_BASIC and
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note If unit information is not available, the value should be interpreted
 *       as time in microseconds.
 */
#define QNN_PROFILE_EVENTTYPE_FINALIZE 300

/**
 * @brief QnnProfile_EventType_t definition to get stats related to execution
 *        of graphs in a context (QnnGraph_execute or QnnGraph_executeAsync).
 *        Basic level might include stats related to execution of entire graphs.
 *        In addition, detailed level can include stats related to individual
 *        nodes in graphs as sub-events.
 *
 * @note execute information maybe available on both QNN_PROFILE_LEVEL_BASIC and
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note If unit information is not available, the value should be interpreted
 *       as time in microseconds.
 */
#define QNN_PROFILE_EVENTTYPE_EXECUTE 400

/**
 * @brief QnnProfile_EventType_t definition to get data related to execution of
 *        an operation. This value can be interpreted appropriately in conjunction
 *        with the unit.
 *
 * @note node specific information is available on QNN_PROFILE_LEVEL_DETAILED level
 *
 * @note This is a sub-event of the QNN_PROFILE_EVENTTYPE_EXECUTE event.
 */
#define QNN_PROFILE_EVENTTYPE_NODE 404

/**
 * @brief QnnProfile_EventType_t definition to get stats related to time spent
 *        waiting in a queue when executing a graph.
 *
 * @note execute enqueue information maybe available on both QNN_PROFILE_LEVEL_BASIC
 *       and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note This is a sub-event of the QNN_PROFILE_EVENTTYPE_EXECUTE event.
 */
#define QNN_PROFILE_EVENTTYPE_EXECUTE_QUEUE_WAIT 405

/**
 * @brief QnnProfile_EventType_t definition to get stats related to time spent
 *        pre-processing in preparation of executing a graph.
 *
 * @note execute preprocess information maybe available on both QNN_PROFILE_LEVEL_BASIC
 *       and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note This is a sub-event of the QNN_PROFILE_EVENTTYPE_EXECUTE event.
 */
#define QNN_PROFILE_EVENTTYPE_EXECUTE_PREPROCESS 406

/**
 * @brief QnnProfile_EventType_t definition to get stats related to time spent
 *        on-device executing a graph.
 *
 * @note execute device information maybe available on both QNN_PROFILE_LEVEL_BASIC
 *       and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note This is a sub-event of the QNN_PROFILE_EVENTTYPE_EXECUTE event.
 */
#define QNN_PROFILE_EVENTTYPE_EXECUTE_DEVICE 407

/**
 * @brief QnnProfile_EventType_t definition to get stats related to time spent
 *        post-processing after execution of a graph.
 *
 * @note execute postprocess information maybe available on both QNN_PROFILE_LEVEL_BASIC
 *       and QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note This is a sub-event of the QNN_PROFILE_EVENTTYPE_EXECUTE event.
 */
#define QNN_PROFILE_EVENTTYPE_EXECUTE_POSTPROCESS 408

/**
 * @brief QnnProfile_EventType_t definition to get stats related to deinit
 *        graphs and free context operation. This profile data captures stats
 *        for QnnContext_free.
 *
 * @note deinit information maybe available on both QNN_PROFILE_LEVEL_BASIC and
 *       QNN_PROFILE_LEVEL_DETAILED levels
 *
 * @note If unit information is not available, the value should be interpreted
 *       as time in microseconds.
 */
#define QNN_PROFILE_EVENTTYPE_DEINIT 500

/**
 * @brief QnnProfile_EventType_t definition to get traces related to graph
 *        preparation and execution steps. This profile data captures stats
 *        for QnnGraph_execute.
 *
 * @note trace information is available on QNN_PROFILE_LEVEL_DETAILED
 *       level only.
 */
#define QNN_PROFILE_EVENTTYPE_TRACE 600

/**
 * @brief QnnProfile_EventType_t definition reserved for each back end to define
 *        and extend
 *
 * @note The client should consult the backend-specific SDK documentation for
 *       information regarding interpretation of unit, value and identifier.
 */
#define QNN_PROFILE_EVENTTYPE_BACKEND 1000

/**
 * @brief Basic QnnProfile_Level_t definition that allows to collect performance
 *        metrics for graph finalization and execution stages.
 */
#define QNN_PROFILE_LEVEL_BASIC 1

/**
 * @brief Detailed QnnProfile_Level_t definition that allows to collect performance
 *        metrics for each operation in the graph
 */
#define QNN_PROFILE_LEVEL_DETAILED 2

/**
 * @brief QnnProfile_Level_t definition reserved for each back end to define and
 *        extend
 */
#define QNN_PROFILE_LEVEL_BACKEND 1000

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement as
 *        time in microseconds
 */
#define QNN_PROFILE_EVENTUNIT_MICROSEC 1

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement as
 *        memory in bytes
 */
#define QNN_PROFILE_EVENTUNIT_BYTES 2

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement as
 *        time in cycles
 */
#define QNN_PROFILE_EVENTUNIT_CYCLES 3

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement as
 *        a count
 */
#define QNN_PROFILE_EVENTUNIT_COUNT 4

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement as
 *        an opaque object
 */
#define QNN_PROFILE_EVENTUNIT_OBJECT 5

/**
 * @brief QnnProfile_EventUnit_t definition to provide profiling measurement with
 *        no unit
 */
#define QNN_PROFILE_EVENTUNIT_NONE 6

/**
 * @brief QnnProfile_EventUnit_t definition reserved for each back end to define
 *        and extend
 */
#define QNN_PROFILE_EVENTUNIT_BACKEND 1000

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief QNN Profile API result / error codes.
 */
typedef enum {
  QNN_PROFILE_MIN_ERROR = QNN_MIN_ERROR_PROFILE,
  ////////////////////////////////////////////

  /// Qnn Profile success
  QNN_PROFILE_NO_ERROR = QNN_SUCCESS,
  /// Backend does not support requested functionality
  QNN_PROFILE_ERROR_UNSUPPORTED = QNN_COMMON_ERROR_NOT_SUPPORTED,
  /// Invalid function argument
  QNN_PROFILE_ERROR_INVALID_ARGUMENT = QNN_COMMON_ERROR_INVALID_ARGUMENT,
  /// General error relating to memory allocation in Profile API
  QNN_PROFILE_ERROR_MEM_ALLOC = QNN_COMMON_ERROR_MEM_ALLOC,
  /// Invalid/NULL QNN profile handle
  QNN_PROFILE_ERROR_INVALID_HANDLE = QNN_MIN_ERROR_PROFILE + 0,
  /// Attempt to free or reconfigure a profile handle that is in-use
  QNN_PROFILE_ERROR_HANDLE_IN_USE = QNN_MIN_ERROR_PROFILE + 1,
  /// Event is incompatible with API
  QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT = QNN_MIN_ERROR_PROFILE + 2,

  ////////////////////////////////////////////
  QNN_PROFILE_MAX_ERROR = QNN_MAX_ERROR_PROFILE,
  // Unused, present to ensure 32 bits.
  QNN_PROFILE_ERROR_UNDEFINED = 0x7FFFFFFF
} QnnProfile_Error_t;

/**
 * @brief Backend defined type for a profiled event such as time_taken, time_start, memory
 */
typedef uint32_t QnnProfile_EventType_t;

/**
 * @brief Represents a profiled event value
 */
typedef uint64_t QnnProfile_EventValue_t;

/**
 * @brief Profile levels supported by each backend
 */
typedef uint32_t QnnProfile_Level_t;

/**
 * @brief ID of a profiling event
 */
typedef uint64_t QnnProfile_EventId_t;

/**
 * @brief Unit of measurement of a profiling event
 */
typedef uint32_t QnnProfile_EventUnit_t;

/**
 * @brief This struct provides event information.
 */
typedef struct {
  /// Type of event
  QnnProfile_EventType_t type;
  /// Unit of measurement for the event
  QnnProfile_EventUnit_t unit;
  /// Value for the event
  QnnProfile_EventValue_t value;
  /// Identifier for the event
  const char* identifier;
} QnnProfile_EventData_t;

// clang-format off
/// QnnProfile_EventData_t initializer macro
#define QNN_PROFILE_EVENT_DATA_INIT \
  {                                 \
    0u,      /*type*/               \
    0u,      /*unit*/               \
    0u,      /*value*/              \
    NULL     /*identifier*/         \
  }
// clang-format on

/**
 * @brief A struct which defines a backend opaque object
 */
typedef struct {
  /// Opaque object
  Qnn_OpaqueObject_t opaqueObject;
  /// Name of the file. Can be NULL.
  const char* fileName;
} QnnProfile_BackendOpaqueObject_t;

// clang-format off
/// QnnProfile_BackendOpaqueObject_t initializer macro
#define QNN_PROFILE_BACKEND_OPAQUE_OBJECT_INIT \
  {                                            \
    QNN_OPAQUE_OBJECT_INIT, /*opaqueObject*/   \
    NULL                    /*fileName*/       \
  }
// clang-format on

/**
 * @brief This struct provides extended event information.
 */
typedef struct {
  /// Type of the event
  QnnProfile_EventType_t type;
  /// Unit of measurement for the event
  QnnProfile_EventUnit_t unit;
  /// Event data
  /// The field used is dependent on the event unit.
  union UNNAMED {
    /// Used for MICROSEC, BYTES, CYCLES, COUNT.
    Qnn_Scalar_t value;
    /// Used for OBJECT.
    QnnProfile_BackendOpaqueObject_t backendOpaqueObject;
  };
  /// Timestamp for the event, represented in microsecond unit.
  uint64_t timestamp;
  /// Identifier for the event. Can be NULL.
  const char* identifier;
} QnnProfile_ExtendedEventDataV1_t;

// clang-format off
/// QnnProfile_ExtendedEventDataV1_t initializer macro
#define QNN_PROFILE_EXTENDED_EVENT_DATA_V1_INIT \
  {                                             \
    0u,               /*type*/                  \
    0u,               /*unit*/                  \
    {                                           \
      QNN_SCALAR_INIT /*value*/                 \
    },                                          \
    0u,               /*timestamp*/             \
    NULL              /*identifier*/            \
  }
// clang-format on

typedef enum {
  QNN_PROFILE_DATA_VERSION_1         = 1,
  QNN_PROFILE_DATA_VERSION_UNDEFINED = 0x7FFFFFFF
} QnnProfile_ExtendedEventDataVersion_t;

typedef struct {
  QnnProfile_ExtendedEventDataVersion_t version;
  union UNNAMED {
    QnnProfile_ExtendedEventDataV1_t v1;
  };
} QnnProfile_ExtendedEventData_t;

// clang-format off
/// QnnProfile_ExtendedEventData_t initializer macro
#define QNN_PROFILE_EXTENDED_EVENT_DATA_INIT         \
  {                                                  \
    QNN_PROFILE_DATA_VERSION_1, /*version*/          \
    {                                                \
      QNN_PROFILE_EXTENDED_EVENT_DATA_V1_INIT /*v1*/ \
    }                                                \
  }
// clang-format on

/**
 * @brief This enum defines profile config options.
 */
typedef enum {
  /// Sets backend custom configs, see backend specific documentation.
  QNN_PROFILE_CONFIG_OPTION_CUSTOM = 0,
  /// This config sets the maximum number of profiling events
  /// that can be stored in the profile handle. Once the maximum
  /// number of events is reached, no more events will be stored.
  /// The absolute maximum number of events is subject to a maximum limit
  /// determined by the backend and available system resources. The default
  /// maximum number of events is backend-specific, refer to SDK documentation.
  QNN_PROFILE_CONFIG_OPTION_MAX_EVENTS = 1,
  /// Set optrace profiling support via enableOptrace flag.
  /// Please note that the trace information is available on QNN_PROFILE_LEVEL_DETAILED level only.
  QNN_PROFILE_CONFIG_OPTION_ENABLE_OPTRACE = 2,
  /// Value selected to ensure 32 bits.
  QNN_PROFILE_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF
} QnnProfile_ConfigOption_t;

/**
 * @brief Profile specific object for custom configuration
 *
 * Please refer to documentation provided by the backend for usage information
 */
typedef void* QnnProfile_CustomConfig_t;

/**
 * @brief This struct provides profile configuration.
 */
typedef struct {
  QnnProfile_ConfigOption_t option;
  union UNNAMED {
    QnnProfile_CustomConfig_t customConfig;
    uint64_t numMaxEvents;
    uint8_t enableOptrace;
  };
} QnnProfile_Config_t;

// clang-format off
/// QnnProfile_Config_t initializer macro
#define QNN_PROFILE_CONFIG_INIT                     \
  {                                                 \
    QNN_PROFILE_CONFIG_OPTION_UNDEFINED, /*option*/ \
    {                                               \
      NULL /*customConfig*/                         \
    }                                               \
  }
// clang-format on

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief Create a handle to a profile object.
 *
 * @param[in] backend A backend handle.
 *
 * @param[in] level Granularity level at which the profile should collect events.
 *
 * @param[out] profile A profile handle.
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: _profile_ is NULL or _level_ is invalid.
 *         - QNN_PROFILE_ERROR_UNSUPPORTED: Profiling is unsupported on a backend.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: Error in allocating memory when creating profile handle
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _backend_ is not a valid handle
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_create(Qnn_BackendHandle_t backend,
                                    QnnProfile_Level_t level,
                                    Qnn_ProfileHandle_t* profile);

/**
 * @brief A function to set/modify configuration options on an already created profile handle.
 *
 * @param[in] profileHandle A profile handle.
 *
 * @param[in] config Pointer to a NULL terminated array of config option pointers. NULL is allowed
 *                   and indicates no config options are provided. All config options have default
 *                   value, in case not provided. If same config option type is provided multiple
 *                   times, the last option value will be used. If a backend cannot support all
 *                   provided configs it will fail.
 *
 * @return Error code:
 *         - QNN_SUCCESS: no error is encountered
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _profileHandle_ is not a valid handle
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: at least one config option is invalid
 *         - QNN_PROFILE_ERROR_HANDLE_IN_USE: when attempting to reconfigure a profile handle
 *         - QNN_PROFILE_ERROR_UNSUPPORTED: Config option is not supported
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_setConfig(Qnn_ProfileHandle_t profileHandle,
                                       const QnnProfile_Config_t** config);

/**
 * @brief Get Qnn profile events collected on the profile handle.
 *
 * @param[in] profile A profile handle.
 *
 * @param[out] profileEventIds Returns handles to Qnn profile events collected on this profile
 *                             object.
 *
 * @param[out] numEvents Number of profile events.
 *
 * @note profileEvents parameter: profile event memory is associated with the profile object and
 *       released on profile object release in QnnProfile_free().
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: _profileEventIds_ or _numEvents_ is NULL.
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _profile_ is not a valid handle.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: error related to memory allocation
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_getEvents(Qnn_ProfileHandle_t profile,
                                       const QnnProfile_EventId_t** profileEventIds,
                                       uint32_t* numEvents);

/**
 * @brief Get Qnn profile event handles nested within this Qnn profile event handle.
 *
 * @param[in] eventId QNN Profile event whose sub events are being queried.
 *
 * @param[out] subEventIds Nested profile events on this event.
 *
 * @param[out] numSubEvents Number of profile events.
 *
 * @note subEventIds parameter: profile event memory is associated with the profile object and
 *       released on profile object release in QnnProfile_free().
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: _subEventIds_ or _numSubEvents_ is NULL.
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _eventId_ does not identify a valid event.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: error related to memory allocation
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_getSubEvents(QnnProfile_EventId_t eventId,
                                          const QnnProfile_EventId_t** subEventIds,
                                          uint32_t* numSubEvents);

/**
 * @brief Query the data associated with this profile event.
 *
 * @param[in] eventId Qnn profile event being queried.
 *
 * @param[out] eventData Event data associated to this event.
 *
 * @note eventData parameter: eventData memory is associated with the profile object and released
 *       on profile object release in QnnProfile_free().
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_UNSUPPORTED: API not supported.
 *         - QNN_PROFILE_ERROR_INCOMPATIBLE_EVENT: _eventData_ is incompatible with the API. Use
 *           QnnProfile_getExtendedEventData instead.
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: _eventData_ is NULL.
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _eventId_ does not identify a valid event.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: error related to memory allocation
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_getEventData(QnnProfile_EventId_t eventId,
                                          QnnProfile_EventData_t* eventData);

/**
 * @brief Query the data associated with this profile extended event.
 *
 * @param[in] eventId Qnn profile extended event being queried.
 *
 * @param[out] eventData Event data associated to this extended event.
 *
 * @note eventData parameter: eventData memory is associated with the profile object and released
 *       on profile object release in QnnProfile_free().
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_UNSUPPORTED: API not supported.
 *         - QNN_PROFILE_ERROR_INVALID_ARGUMENT: _eventData_ is NULL.
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _eventId_ does not identify a valid event.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: error related to memory allocation
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_getExtendedEventData(QnnProfile_EventId_t eventId,
                                                  QnnProfile_ExtendedEventData_t* eventData);

/**
 * @brief Free memory associated with the profile handle.
 *        All associated QnnProfile_EventId_t event handles are implicitly freed.
 *
 * @param[in] profile Handle to be freed.
 *
 * @note Releasing the profile handle invalidates the memory returned via calls on this handle such
 *       as QnnProfile_getEvents(), QnnProfile_getSubEvents(), QnnProfile_getEventData(),
 *       QnnProfile_getExtendedEventData(), etc.
 *
 * @note The profile handle cannot be freed when it is bound to another API component or
 *       in use by an API call.
 *
 * @return Error code
 *         - QNN_SUCCESS: No error encountered
 *         - QNN_PROFILE_ERROR_INVALID_HANDLE: _profile_ is not a valid handle.
 *         - QNN_PROFILE_ERROR_MEM_ALLOC: error related to memory de-allocation
 *         - QNN_PROFILE_ERROR_HANDLE_IN_USE: _profile_ is in-use and cannot be freed.
 *
 * @note Use corresponding API through QnnInterface_t.
 */
QNN_API
Qnn_ErrorHandle_t QnnProfile_free(Qnn_ProfileHandle_t profile);

#ifdef __cplusplus
}
#endif

#endif  // QNN_PROFILE_H
