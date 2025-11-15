@echo off
setlocal enabledelayedexpansion

echo Running Cactus test suite...
echo ============================

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

set "WEIGHTS_DIR=%PROJECT_ROOT%\weights\lfm2-1.2B"
if not exist "%WEIGHTS_DIR%\config.txt" (
    echo.
    echo Weights not found. Generating weights...
    echo =============================================
    cd /d "%PROJECT_ROOT%"
    where python3 >nul 2>&1
    if !errorlevel!==0 (
        echo Running: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        if !errorlevel!==0 (
            echo Successfully generated Weights
        ) else (
            echo Warning: Failed to generate Weights. Tests may fail.
            echo Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
        )
    ) else (
        echo Warning: Python3 not found. Cannot generate weights automatically.
        echo Please run manually: python3 tools/convert_hf.py LiquidAI/LFM2-1.2B weights/lfm2-1.2B/ --precision INT8
    )
) else (
    echo.
    echo Weights found at %WEIGHTS_DIR%
)

echo.
echo Step 1: Building Cactus library...
cd /d "%PROJECT_ROOT%"
call cactus\build.bat
if !errorlevel! neq 0 (
    echo Failed to build cactus library
    exit /b 1
)

echo.
echo Step 2: Building tests...
cd /d "%PROJECT_ROOT%\tests"

if exist build rmdir /s /q build
mkdir build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF >nul 2>&1
if !errorlevel! neq 0 (
    echo Failed to configure tests
    exit /b 1
)

cmake --build . --config Release
if !errorlevel! neq 0 (
    echo Failed to build tests
    exit /b 1
)

echo.
echo Step 3: Running tests...
echo ------------------------

echo Discovering test executables...
set "test_count=0"
for %%f in (Release\test_*.exe Debug\test_*.exe test_*.exe) do (
    if exist "%%f" (
        set /a test_count+=1
        set "test_!test_count!=%%f"
    )
)

if !test_count!==0 (
    echo No test executables found!
    exit /b 1
)

echo Found !test_count! test executable(s)

for /l %%i in (1,1,!test_count!) do (
    "!test_%%i!"
)
