//
//  CactusSDK-Bridging-Header.h
//  Cactus
//
//  Created by AgentZero.
//  Copyright © 2024 Cactus Compute. All rights reserved.
//
//  Use this file to import your target's public C headers that you would like to expose to Swift.
//

#ifndef CactusSDK_Bridging_Header_h
#define CactusSDK_Bridging_Header_h

// Import for AVFoundation if any C struct/const from it were needed by C code, unlikely for this use case.
// #import <AVFoundation/AVFoundation.h>

// This is where you would import the C header file(s) from your C++ library (cactus.xcframework)
// that declare the FFI functions for STT.
// For example, if your FFI functions are in "cactus_stt_ffi.h" or a general "cactus.h"
// which is made available as a public header by the framework:

// Option 1: If headers are directly available in the framework's public headers
// #import <cactus/cactus_stt_ffi.h> // Assuming 'cactus' is the module name of the XCFramework
                                     // and 'cactus_stt_ffi.h' is a public header.

// Option 2: If headers are at the root of the framework's include path (less common for XCFrameworks)
// #import "cactus_stt_ffi.h"

// For now, this is a placeholder. The actual import path depends on how `cactus.xcframework`
// exposes its C headers. If the framework uses a module map (`module.modulemap`),
// explicit imports here might not even be needed for C functions if they are part of the module.
// However, for direct C function calls not part of a module, this bridging header is key.

// Placeholder FFI function declarations (if not in an imported header)
// It's much better to have these in a proper C header within the XCFramework.
// These are just to illustrate what Swift would need to see.
// extern "C" { // Not needed in bridging header, just in C/C++ headers
// int cactus_stt_init_ffi(const char* model_path);
// const char* cactus_stt_process_file_ffi(const char* file_path);
// void cactus_stt_release_ffi();
// void cactus_stt_free_string_ffi(const char* str);
// }


#endif /* CactusSDK_Bridging_Header_h */
