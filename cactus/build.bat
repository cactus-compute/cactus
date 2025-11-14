@echo off
setlocal

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

echo Building Cactus library...

cd /d "%~dp0"

if exist build rmdir /s /q build

mkdir build
cd build

cmake .. -DCMAKE_RULE_MESSAGES=OFF -DCMAKE_VERBOSE_MAKEFILE=OFF >nul 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build . --config Release
if %errorlevel% neq 0 exit /b %errorlevel%

echo Cactus library built successfully!
echo Library location: %cd%\lib\libcactus.a
