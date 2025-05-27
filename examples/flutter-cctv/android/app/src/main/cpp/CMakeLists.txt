cmake_minimum_required(VERSION 3.10)
project(flutter_chat_app_dummy_code)

# Create a dummy C++ source file on the fly
# This ensures the target has actual source code to compile.
file(WRITE ${CMAKE_BINARY_DIR}/app_dummy_source.cpp 
"// Dummy C++ file\nvoid app_specific_dummy_function() {}\n"
)

add_library(
    app_specific_dummy_library # A unique library name for the app
    SHARED
    ${CMAKE_BINARY_DIR}/app_dummy_source.cpp
)

# This CMakeLists.txt is intentionally trying to build a harmless, uniquely named
# dummy library for the app module. The goal is to satisfy any implicit
# native build requirements of the Flutter/Android build process for the app,
# without it interfering with 'libcactus.so' which should be provided by
# the 'cactus-flutter' dependent module. 