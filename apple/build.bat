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
set "ROOT_DIR=%SCRIPT_DIR%.."
set "APPLE_DIR=%ROOT_DIR%\apple"

if "%CMAKE_BUILD_TYPE%"=="" set "CMAKE_BUILD_TYPE=Release"
if "%BUILD_STATIC%"=="" set "BUILD_STATIC=true"
if "%BUILD_XCFRAMEWORK%"=="" set "BUILD_XCFRAMEWORK=true"

where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: cmake not found, please install it
    exit /b 1
)

where xcodebuild >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Xcode command line tools not found
    echo Install with: xcode-select --install
    exit /b 1
)

set "n_cpu=%NUMBER_OF_PROCESSORS%"
if "%n_cpu%"=="" set "n_cpu=4"

echo Building Cactus for Apple platforms...
echo Build type: %CMAKE_BUILD_TYPE%
echo Using %n_cpu% CPU cores
echo Static library: %BUILD_STATIC%
echo XCFramework: %BUILD_XCFRAMEWORK%

set "start_time=%time%"

if "%BUILD_STATIC%"=="true" (
    call :build_static_library
    if !errorlevel! neq 0 exit /b !errorlevel!
)

if "%BUILD_XCFRAMEWORK%"=="true" (
    call :build_ios_xcframework
    if !errorlevel! neq 0 exit /b !errorlevel!
    call :build_macos_xcframework
    if !errorlevel! neq 0 exit /b !errorlevel!
)

set "end_time=%time%"

echo.
echo Build complete!

if "%BUILD_STATIC%"=="true" (
    if exist "%APPLE_DIR%\build-static-device" rmdir /s /q "%APPLE_DIR%\build-static-device"
    if exist "%APPLE_DIR%\build-static-simulator" rmdir /s /q "%APPLE_DIR%\build-static-simulator"
    if exist "%APPLE_DIR%\build-static-macos" rmdir /s /q "%APPLE_DIR%\build-static-macos"
    echo Static libraries:
    echo   Device: %APPLE_DIR%\libcactus-device.a
    echo   Simulator: %APPLE_DIR%\libcactus-simulator.a
)

if "%BUILD_XCFRAMEWORK%"=="true" (
    echo XCFrameworks:
    echo   iOS: %APPLE_DIR%\cactus-ios.xcframework
    echo   macOS: %APPLE_DIR%\cactus-macos.xcframework
)

exit /b 0

:build_static_library
echo Building static library for iOS device...
set "BUILD_DIR=%APPLE_DIR%\build-static-device"

for /f "delims=" %%i in ('xcrun --sdk iphoneos --show-sdk-path') do set "IOS_SDK_PATH=%%i"
if "%IOS_SDK_PATH%"=="" (
    echo Error: iOS SDK not found. Make sure Xcode is installed.
    exit /b 1
)
if not exist "%IOS_SDK_PATH%" (
    echo Error: iOS SDK not found. Make sure Xcode is installed.
    exit /b 1
)

echo Using iOS SDK: %IOS_SDK_PATH%

cmake -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_OSX_SYSROOT="%IOS_SDK_PATH%" -DCMAKE_BUILD_TYPE="%CMAKE_BUILD_TYPE%" -DBUILD_SHARED_LIBS=OFF -S "%APPLE_DIR%" -B "%BUILD_DIR%"
if !errorlevel! neq 0 exit /b !errorlevel!

cmake --build "%BUILD_DIR%" --config "%CMAKE_BUILD_TYPE%" -j %n_cpu%
if !errorlevel! neq 0 exit /b !errorlevel!

if not exist "%APPLE_DIR%" mkdir "%APPLE_DIR%"
copy "%BUILD_DIR%\libcactus.a" "%APPLE_DIR%\libcactus-device.a" >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: Could not find device libcactus.a
    exit /b 1
)

echo Device static library built: %APPLE_DIR%\libcactus-device.a

echo Building static library for iOS simulator...
set "BUILD_DIR_SIM=%APPLE_DIR%\build-static-simulator"

for /f "delims=" %%i in ('xcrun --sdk iphonesimulator --show-sdk-path') do set "IOS_SIM_SDK_PATH=%%i"
if "%IOS_SIM_SDK_PATH%"=="" (
    echo Error: iOS Simulator SDK not found. Make sure Xcode is installed.
    exit /b 1
)
if not exist "%IOS_SIM_SDK_PATH%" (
    echo Error: iOS Simulator SDK not found. Make sure Xcode is installed.
    exit /b 1
)

echo Using iOS Simulator SDK: %IOS_SIM_SDK_PATH%

cmake -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_OSX_SYSROOT="%IOS_SIM_SDK_PATH%" -DCMAKE_BUILD_TYPE="%CMAKE_BUILD_TYPE%" -DBUILD_SHARED_LIBS=OFF -S "%APPLE_DIR%" -B "%BUILD_DIR_SIM%"
if !errorlevel! neq 0 exit /b !errorlevel!

cmake --build "%BUILD_DIR_SIM%" --config "%CMAKE_BUILD_TYPE%" -j %n_cpu%
if !errorlevel! neq 0 exit /b !errorlevel!

copy "%BUILD_DIR_SIM%\libcactus.a" "%APPLE_DIR%\libcactus-simulator.a" >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: Could not find simulator libcactus.a
    exit /b 1
)

echo Simulator static library built: %APPLE_DIR%\libcactus-simulator.a
exit /b 0

:build_framework
echo Building framework for %~4...
cd /d "%~5"

cmake -S "%ROOT_DIR%\apple" -B . -GXcode -DCMAKE_SYSTEM_NAME=%~1 -DCMAKE_OSX_ARCHITECTURES=%~2 -DCMAKE_OSX_SYSROOT=%~3 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 -DCMAKE_BUILD_TYPE="%CMAKE_BUILD_TYPE%" -DBUILD_SHARED_LIBS=ON -DCMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH=NO -DCMAKE_IOS_INSTALL_COMBINED=YES
if !errorlevel! neq 0 exit /b !errorlevel!

cmake --build . --config "%CMAKE_BUILD_TYPE%" -j %n_cpu%
if !errorlevel! neq 0 exit /b !errorlevel!

set "DEST_DIR=%ROOT_DIR%\apple\%~6\%~4"

set "FRAMEWORK_SRC="
if exist "%CMAKE_BUILD_TYPE%-%~3\cactus.framework" (
    set "FRAMEWORK_SRC=%CMAKE_BUILD_TYPE%-%~3\cactus.framework"
) else if exist "%CMAKE_BUILD_TYPE%\cactus.framework" (
    set "FRAMEWORK_SRC=%CMAKE_BUILD_TYPE%\cactus.framework"
) else (
    for /f "delims=" %%f in ('dir /s /b /ad cactus.framework 2^>nul') do (
        set "FRAMEWORK_SRC=%%f"
        goto :found_framework
    )
)

:found_framework
set "FRAMEWORK_DEST=%DEST_DIR%\cactus.framework"

if exist "%DEST_DIR%" rmdir /s /q "%DEST_DIR%"
mkdir "%DEST_DIR%"

if not "%FRAMEWORK_SRC%"=="" (
    if exist "%FRAMEWORK_SRC%" (
        xcopy /E /I /Q /Y "%FRAMEWORK_SRC%" "%FRAMEWORK_DEST%" >nul
        echo Framework copied from %FRAMEWORK_SRC% to %FRAMEWORK_DEST%
    ) else (
        echo Error: Framework not found in build directory
        echo Available files:
        dir /s /b *.framework libcactus* 2>nul
        exit /b 1
    )
) else (
    echo Error: Framework not found in build directory
    echo Available files:
    dir /s /b *.framework libcactus* 2>nul
    exit /b 1
)

call :cp_headers %~6 %~4

cd /d "%ROOT_DIR%"
exit /b 0

:cp_headers
if not exist "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers" mkdir "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers"
copy "%ROOT_DIR%\cactus\ffi\*.h" "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers\" >nul 2>&1
copy "%ROOT_DIR%\cactus\engine\*.h" "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers\" >nul 2>&1
copy "%ROOT_DIR%\cactus\graph\*.h" "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers\" >nul 2>&1
copy "%ROOT_DIR%\cactus\kernel\*.h" "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers\" >nul 2>&1
copy "%ROOT_DIR%\cactus\*.h" "%ROOT_DIR%\apple\%~1\%~2\cactus.framework\Headers\" >nul 2>&1
exit /b 0

:build_ios_xcframework
echo Building iOS XCFramework...

if exist "%ROOT_DIR%\apple\cactus-ios.xcframework" rmdir /s /q "%ROOT_DIR%\apple\cactus-ios.xcframework"
if exist "%ROOT_DIR%\apple\build-ios" rmdir /s /q "%ROOT_DIR%\apple\build-ios"
if exist "%ROOT_DIR%\apple\build-ios-simulator" rmdir /s /q "%ROOT_DIR%\apple\build-ios-simulator"
mkdir "%ROOT_DIR%\apple\build-ios"
mkdir "%ROOT_DIR%\apple\build-ios-simulator"

call :build_framework "iOS" "arm64" "iphoneos" "ios-arm64" "%ROOT_DIR%\apple\build-ios" "cactus-ios.xcframework"
if !errorlevel! neq 0 exit /b !errorlevel!

call :build_framework "iOS" "arm64" "iphonesimulator" "ios-arm64-simulator" "%ROOT_DIR%\apple\build-ios-simulator" "cactus-ios.xcframework"
if !errorlevel! neq 0 exit /b !errorlevel!

call :create_ios_xcframework_info_plist

if exist "%ROOT_DIR%\apple\build-ios" rmdir /s /q "%ROOT_DIR%\apple\build-ios"
if exist "%ROOT_DIR%\apple\build-ios-simulator" rmdir /s /q "%ROOT_DIR%\apple\build-ios-simulator"

echo iOS XCFramework built: %ROOT_DIR%\apple\cactus-ios.xcframework
exit /b 0

:build_macos_xcframework
echo Building macOS XCFramework...

if exist "%ROOT_DIR%\apple\cactus-macos.xcframework" rmdir /s /q "%ROOT_DIR%\apple\cactus-macos.xcframework"
if exist "%ROOT_DIR%\apple\build-macos" rmdir /s /q "%ROOT_DIR%\apple\build-macos"
mkdir "%ROOT_DIR%\apple\build-macos"

call :build_framework "Darwin" "arm64" "macosx" "macos-arm64" "%ROOT_DIR%\apple\build-macos" "cactus-macos.xcframework"
if !errorlevel! neq 0 exit /b !errorlevel!

call :create_macos_xcframework_info_plist

if exist "%ROOT_DIR%\apple\build-macos" rmdir /s /q "%ROOT_DIR%\apple\build-macos"

echo macOS XCFramework built: %ROOT_DIR%\apple\cactus-macos.xcframework
exit /b 0

:create_ios_xcframework_info_plist
(
echo ^<?xml version="1.0" encoding="UTF-8"?^>
echo ^<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"^>
echo ^<plist version="1.0"^>
echo ^<dict^>
echo 	^<key^>AvailableLibraries^</key^>
echo 	^<array^>
echo 		^<dict^>
echo 			^<key^>LibraryIdentifier^</key^>
echo 			^<string^>ios-arm64^</string^>
echo 			^<key^>LibraryPath^</key^>
echo 			^<string^>cactus.framework^</string^>
echo 			^<key^>SupportedArchitectures^</key^>
echo 			^<array^>
echo 				^<string^>arm64^</string^>
echo 			^</array^>
echo 			^<key^>SupportedPlatform^</key^>
echo 			^<string^>ios^</string^>
echo 		^</dict^>
echo 		^<dict^>
echo 			^<key^>LibraryIdentifier^</key^>
echo 			^<string^>ios-arm64-simulator^</string^>
echo 			^<key^>LibraryPath^</key^>
echo 			^<string^>cactus.framework^</string^>
echo 			^<key^>SupportedArchitectures^</key^>
echo 			^<array^>
echo 				^<string^>arm64^</string^>
echo 			^</array^>
echo 			^<key^>SupportedPlatform^</key^>
echo 			^<string^>ios^</string^>
echo 			^<key^>SupportedPlatformVariant^</key^>
echo 			^<string^>simulator^</string^>
echo 		^</dict^>
echo 	^</array^>
echo 	^<key^>CFBundlePackageType^</key^>
echo 	^<string^>XFWK^</string^>
echo 	^<key^>XCFrameworkFormatVersion^</key^>
echo 	^<string^>1.0^</string^>
echo ^</dict^>
echo ^</plist^>
) > "%ROOT_DIR%\apple\cactus-ios.xcframework\Info.plist"
exit /b 0

:create_macos_xcframework_info_plist
(
echo ^<?xml version="1.0" encoding="UTF-8"?^>
echo ^<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd"^>
echo ^<plist version="1.0"^>
echo ^<dict^>
echo 	^<key^>AvailableLibraries^</key^>
echo 	^<array^>
echo 		^<dict^>
echo 			^<key^>LibraryIdentifier^</key^>
echo 			^<string^>macos-arm64^</string^>
echo 			^<key^>LibraryPath^</key^>
echo 			^<string^>cactus.framework^</string^>
echo 			^<key^>SupportedArchitectures^</key^>
echo 			^<array^>
echo 				^<string^>arm64^</string^>
echo 			^</array^>
echo 			^<key^>SupportedPlatform^</key^>
echo 			^<string^>macos^</string^>
echo 		^</dict^>
echo 	^</array^>
echo 	^<key^>CFBundlePackageType^</key^>
echo 	^<string^>XFWK^</string^>
echo 	^<key^>XCFrameworkFormatVersion^</key^>
echo 	^<string^>1.0^</string^>
echo ^</dict^>
echo ^</plist^>
) > "%ROOT_DIR%\apple\cactus-macos.xcframework\Info.plist"
exit /b 0
