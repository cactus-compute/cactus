@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: CMake not found.
    echo Install via: winget install Kitware.CMake
    exit /b 1
)

where clang++ >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: clang++ not found.
    echo Install LLVM via: winget install LLVM.LLVM
    exit /b 1
)

where ninja >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: ninja not found.
    echo Install Ninja via: winget install Ninja-build.Ninja
    exit /b 1
)

echo Building Cactus library...
cmake -B "%PROJECT_ROOT%\cactus\build" -S "%PROJECT_ROOT%\cactus" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=clang++
if %ERRORLEVEL% neq 0 (
    echo CMake configure failed
    exit /b %ERRORLEVEL%
)

cmake --build "%PROJECT_ROOT%\cactus\build" --parallel
if %ERRORLEVEL% neq 0 (
    echo Cactus library build failed
    exit /b %ERRORLEVEL%
)

echo Building chat binary...
cmake -B "%PROJECT_ROOT%\tests\build" -S "%PROJECT_ROOT%\tests" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CXX_COMPILER=clang++
if %ERRORLEVEL% neq 0 (
    echo Tests CMake configure failed
    exit /b %ERRORLEVEL%
)

cmake --build "%PROJECT_ROOT%\tests\build" --target chat --parallel
if %ERRORLEVEL% neq 0 (
    echo Chat binary build failed
    exit /b %ERRORLEVEL%
)

echo.
echo Build complete!
echo Library: %PROJECT_ROOT%\cactus\build\libcactus.a
echo Binary:  %PROJECT_ROOT%\tests\build\chat.exe
