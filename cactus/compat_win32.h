#pragma once
#ifdef _WIN32

#include <stddef.h>
#include <string.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Windows headers define ERROR, BOOL, min, max etc. which conflict with cactus identifiers
#undef ERROR
#undef BOOL
#undef TRUE
#undef FALSE

#define MADV_NORMAL     0
#define MADV_RANDOM     1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED   3
#define MADV_DONTNEED   4

inline int madvise(void* /*addr*/, size_t /*length*/, int /*advice*/) { return 0; }

// 2-arg mkdir(path, mode) — Windows _mkdir ignores mode
#include <direct.h>
inline int mkdir(const char* path, int /*mode*/) { return _mkdir(path); }

inline int fsync(int fd) { return _commit(fd); }

// Stub for sys/utsname.h (POSIX-only)
struct utsname {
    char sysname[256];
    char nodename[256];
    char release[256];
    char version[256];
    char machine[256];
};
inline int uname(struct utsname* u) {
    memset(u, 0, sizeof(*u));
    strncpy(u->sysname, "Windows", sizeof(u->sysname) - 1);
    strncpy(u->machine, "ARM64", sizeof(u->machine) - 1);
    strncpy(u->release, "Unknown", sizeof(u->release) - 1);
    strncpy(u->version, "Unknown", sizeof(u->version) - 1);
    return 0;
}

#endif
