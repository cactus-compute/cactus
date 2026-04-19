// Force-included before other headers during MSVC ARM64 host build.
// Include log.h first so it sets up function templates, then immediately
// override the broken log macros that use the FILE_BASENAME ":" ... literal
// concat pattern (which MSVC rejects inside template instantiations).

#pragma once

#include "HTP/core/log.h"

#undef debuglog
#define debuglog(...)   ((void)0)
#undef okaylog
#define okaylog(...)    ((void)0)
#undef _debuglog
#define _debuglog(...)  ((void)0)
#undef logmsg
#define logmsg(...)     ((void)0)
#undef errlog
// errlog is consumed as an expression of type GraphStatus; give it a fatal value.
#define errlog(...)     (GraphStatus::ErrorFatal)
