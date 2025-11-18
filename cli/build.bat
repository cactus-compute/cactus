@echo off
setlocal enabledelayedexpansion

echo Building Cactus chat...
echo =======================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

set "WEIGHTS_DIR=%PROJECT_ROOT%\weights\lfm2-1.2B"
if not exist "%WEIGHTS_DIR%\config.txt" (
    echo.
    echo LFM2 weights not found. Generating weights...
    echo =============================================
    cd /d "%PROJECT_ROOT%"
    where python3 >nul 2>&1
    if !errorlevel!==0 (
        echo Running: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        python3 -c "import numpy, torch, transformers" >nul 2>&1
        if !errorlevel! neq 0 (
            echo Warning: Required Python packages not found. Make sure to set up your env in accordance with the README.
            exit /b 1
        )
        python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        if !errorlevel!==0 (
            echo Successfully generated Weights
        ) else (
            echo Warning: Failed to generate Weights. Tests may fail.
            echo Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        )
    ) else (
        where python >nul 2>&1
        if !errorlevel!==0 (
            echo Running: python tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
            python -c "import numpy, torch, transformers" >nul 2>&1
            if !errorlevel! neq 0 (
                echo Warning: Required Python packages not found. Make sure to set up your env in accordance with the README.
                exit /b 1
            )
            python tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
            if !errorlevel!==0 (
                echo Successfully generated Weights
            ) else (
                echo Warning: Failed to generate Weights. Tests may fail.
                echo Please run manually: python tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
            )
        ) else (
            echo Warning: Python not found. Cannot generate weights automatically.
            echo Please run manually: python tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        )
    )
) else (
    echo.
    echo LFM2 weights found at %WEIGHTS_DIR%
)

set "ROOT_DIR=%SCRIPT_DIR%.."
set "BUILD_DIR=%SCRIPT_DIR%build"

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cd /d "%ROOT_DIR%\cactus"
if not exist "build\libcactus.a" (
    if not exist "build\cactus.lib" (
        echo Cactus library not found. Building...
        call build.bat
        if !errorlevel! neq 0 (
            echo Failed to build cactus library
            exit /b 1
        )
    )
)

cd /d "%BUILD_DIR%"

echo Compiling chat.cpp...

REM Check for Visual Studio compiler
where cl >nul 2>&1
if !errorlevel!==0 (
    REM Use MSVC
    cl /std:c++17 /O2 /EHsc ^
        /I"%ROOT_DIR%" ^
        "%SCRIPT_DIR%chat.cpp" ^
        "%ROOT_DIR%\cactus\build\cactus.lib" ^
        /Fe:chat.exe
    if !errorlevel! neq 0 (
        echo Failed to compile with MSVC
        exit /b 1
    )
) else (
    REM Try MinGW
    where g++ >nul 2>&1
    if !errorlevel!==0 (
        g++ -std=c++17 -O3 ^
            -I"%ROOT_DIR%" ^
            "%SCRIPT_DIR%chat.cpp" ^
            "%ROOT_DIR%\cactus\build\libcactus.a" ^
            -o chat.exe ^
            -pthread
        if !errorlevel! neq 0 (
            echo Failed to compile with MinGW
            exit /b 1
        )
    ) else (
        echo Error: No C++ compiler found. Please install Visual Studio or MinGW.
        exit /b 1
    )
)

echo Build complete: %BUILD_DIR%\chat.exe
echo.

cls
echo Usage: %BUILD_DIR%\chat.exe ^<model_path^>
echo.

"%BUILD_DIR%\chat.exe" "%PROJECT_ROOT%\weights\lfm2-1.2B"