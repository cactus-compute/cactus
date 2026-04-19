@echo off
setlocal
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -host_arch=arm64 -arch=arm64 -no_logo
if errorlevel 1 (
  echo VsDevCmd failed
  exit /b 1
)
set QNN_INC=C:\Qualcomm\AIStack\QAIRT\2.31.0.250130\include\QNN
set HEXNATIVE=C:\Qualcomm\Hexagon_SDK\6.4.0.2\tools\HEXAGON_Tools\19.0.04\Tools\libnative\include
mkdir build-winarm64 2>nul
cd /d %~dp0
cl.exe /c /Zi /MDd /std:c++17 /EHsc ^
  /I"%QNN_INC%" ^
  /I"%HEXNATIVE%" ^
  /I. ^
  /D__HVXDBL__ /DUSE_OS_WIN32 /DTHIS_PKG_NAME=CactusSquarePackage ^
  /FI"log_stub.h" ^
  /Fo"build-winarm64\\" ^
  src\CactusSquarePackageInterface.cpp ^
  src\ops\Square.cpp
if errorlevel 1 (echo compile exit=%ERRORLEVEL% & exit /b %ERRORLEVEL%)

link.exe /nologo /DLL ^
  /OUT:build-winarm64\QnnCactusSquarePackage.dll ^
  /MACHINE:ARM64 ^
  /IMPLIB:build-winarm64\QnnCactusSquarePackage.lib ^
  build-winarm64\CactusSquarePackageInterface.obj ^
  build-winarm64\Square.obj
echo link exit=%ERRORLEVEL%
endlocal
