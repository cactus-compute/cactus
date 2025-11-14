@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

echo Cleaning all build artifacts from Cactus project...
echo Project root: %PROJECT_ROOT%
echo.

if exist "%PROJECT_ROOT%\cactus\build" (
    echo Removing: %PROJECT_ROOT%\cactus\build
    rmdir /s /q "%PROJECT_ROOT%\cactus\build"
) else (
    echo Not found: %PROJECT_ROOT%\cactus\build
)

if exist "%PROJECT_ROOT%\android\build" (
    echo Removing: %PROJECT_ROOT%\android\build
    rmdir /s /q "%PROJECT_ROOT%\android\build"
) else (
    echo Not found: %PROJECT_ROOT%\android\build
)

if exist "%PROJECT_ROOT%\android\libs" (
    echo Removing: %PROJECT_ROOT%\android\libs
    rmdir /s /q "%PROJECT_ROOT%\android\libs"
) else (
    echo Not found: %PROJECT_ROOT%\android\libs
)

if exist "%PROJECT_ROOT%\android\arm64-v8a" (
    echo Removing: %PROJECT_ROOT%\android\arm64-v8a
    rmdir /s /q "%PROJECT_ROOT%\android\arm64-v8a"
) else (
    echo Not found: %PROJECT_ROOT%\android\arm64-v8a
)

if exist "%PROJECT_ROOT%\apple\build" (
    echo Removing: %PROJECT_ROOT%\apple\build
    rmdir /s /q "%PROJECT_ROOT%\apple\build"
) else (
    echo Not found: %PROJECT_ROOT%\apple\build
)

if exist "%PROJECT_ROOT%\apple\libcactus.a" (
    echo Removing: %PROJECT_ROOT%\apple\libcactus.a
    del /q "%PROJECT_ROOT%\apple\libcactus.a"
) else (
    echo Not found: %PROJECT_ROOT%\apple\libcactus.a
)

if exist "%PROJECT_ROOT%\tests\build" (
    echo Removing: %PROJECT_ROOT%\tests\build
    rmdir /s /q "%PROJECT_ROOT%\tests\build"
) else (
    echo Not found: %PROJECT_ROOT%\tests\build
)

echo.
echo Removing compiled libraries and frameworks...

set "FOUND_SO=0"
for /r "%PROJECT_ROOT%" %%f in (*.so) do (
    del /q "%%f" >nul 2>&1
    set "FOUND_SO=1"
)
if !FOUND_SO!==1 (
    echo Removed .so files
) else (
    echo No .so files found
)

set "FOUND_A=0"
for /r "%PROJECT_ROOT%" %%f in (*.a) do (
    del /q "%%f" >nul 2>&1
    set "FOUND_A=1"
)
if !FOUND_A!==1 (
    echo Removed .a files
) else (
    echo No .a files found
)

set "FOUND_XCFRAMEWORK=0"
for /d /r "%PROJECT_ROOT%" %%d in (*.xcframework) do (
    if exist "%%d" (
        rmdir /s /q "%%d" >nul 2>&1
        set "FOUND_XCFRAMEWORK=1"
    )
)
if !FOUND_XCFRAMEWORK!==1 (
    echo Removed .xcframework directories
) else (
    echo No .xcframework directories found
)

echo.
echo Clean complete!
echo All build artifacts have been removed.
