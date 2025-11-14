@echo off
setlocal enabledelayedexpansion

rem Ensure console uses UTF-8 and try to auto-install MSYS2 package if pacman is available
chcp.com 65001 >nul 2>&1

where pacman >nul 2>&1
if %ERRORLEVEL%==0 (
    echo Found pacman; checking for mingw-w64-clang-aarch64-mman-win32...
    pacman -Q mingw-w64-clang-aarch64-mman-win32 >nul 2>&1 || (
        echo Installing mingw-w64-clang-aarch64-mman-win32 via pacman...
        pacman -S --noconfirm mingw-w64-clang-aarch64-mman-win32 || (
            echo Warning: pacman failed to install mingw-w64-clang-aarch64-mman-win32.
            echo Please install it manually if mmap support is required.
        )
    )
) else (
    echo pacman not found; skipping automatic install of mingw-w64-clang-aarch64-mman-win32.
    echo If you need mmap support on Windows/MSYS2, run:
    echo   pacman -S mingw-w64-clang-aarch64-mman-win32
)

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "ANDROID_DIR=%PROJECT_ROOT%\android"

if "%ANDROID_PLATFORM%"=="" set "ANDROID_PLATFORM=android-21"
if "%CMAKE_BUILD_TYPE%"=="" set "CMAKE_BUILD_TYPE=Release"
set "BUILD_DIR=%ANDROID_DIR%\build"

if "%ANDROID_NDK_HOME%"=="" (
    if not "%ANDROID_HOME%"=="" (
        for /f "delims=" %%d in ('dir /b /ad /o-n "%ANDROID_HOME%\ndk\*" 2^>nul') do (
            set "ANDROID_NDK_HOME=%ANDROID_HOME%\ndk\%%d"
            goto :ndk_found
        )
    )
    if exist "%HOME%\Library\Android\sdk\ndk" (
        for /f "delims=" %%d in ('dir /b /ad /o-n "%HOME%\Library\Android\sdk\ndk\*" 2^>nul') do (
            set "ANDROID_NDK_HOME=%HOME%\Library\Android\sdk\ndk\%%d"
            goto :ndk_found
        )
    )
)

:ndk_found
if "%ANDROID_NDK_HOME%"=="" (
    echo Error: Android NDK not found.
    echo Set ANDROID_NDK_HOME or install NDK via Android SDK Manager
    exit /b 1
)

if not exist "%ANDROID_NDK_HOME%" (
    echo Error: Android NDK not found.
    echo Set ANDROID_NDK_HOME or install NDK via Android SDK Manager
    exit /b 1
)

echo Using NDK: %ANDROID_NDK_HOME%
set "CMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_HOME%\build\cmake\android.toolchain.cmake"

where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: cmake not found, please install it
    exit /b 1
)

set "n_cpu=%NUMBER_OF_PROCESSORS%"
if "%n_cpu%"=="" set "n_cpu=4"

set "ABI=arm64-v8a"

echo Building Cactus for Android (%ABI%)...
echo Build type: %CMAKE_BUILD_TYPE%
echo Using %n_cpu% CPU cores
echo Android CMakeLists.txt: %ANDROID_DIR%\CMakeLists.txt

cmake -DCMAKE_TOOLCHAIN_FILE="%CMAKE_TOOLCHAIN_FILE%" -DANDROID_ABI="%ABI%" -DANDROID_PLATFORM="%ANDROID_PLATFORM%" -DCMAKE_BUILD_TYPE="%CMAKE_BUILD_TYPE%" -S "%ANDROID_DIR%" -B "%BUILD_DIR%"
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build "%BUILD_DIR%" --config "%CMAKE_BUILD_TYPE%" -j %n_cpu%
if %errorlevel% neq 0 exit /b %errorlevel%

copy "%BUILD_DIR%\lib\libcactus.so" "%ANDROID_DIR%\" >nul 2>&1
if %errorlevel% neq 0 (
    copy "%BUILD_DIR%\libcactus.so" "%ANDROID_DIR%" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Error: Could not find libcactus.so
        exit /b 1
    )
)

copy "%BUILD_DIR%\lib\libcactus_static.a" "%ANDROID_DIR%\libcactus.a" >nul 2>&1
if %errorlevel% neq 0 (
    copy "%BUILD_DIR%\libcactus_static.a" "%ANDROID_DIR%\libcactus.a" >nul 2>&1
    if !errorlevel! neq 0 (
        echo Warning: Could not find libcactus_static.a
    )
)

echo Build complete!
echo Shared library location: %ANDROID_DIR%\libcactus.so
echo Static library location: %ANDROID_DIR%\libcactus.a
