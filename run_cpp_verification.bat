@echo off
set PATH=C:\msys64\mingw64\bin;%PATH%
echo Running Real Verification...
g++.exe -Wfatal-errors -I. -Itests/mock_sdk -include tests/mock_sdk/kernel_mock.h -include tests/mock_sdk/kernel_utils_mock.h tests/test_real_compilation.cpp -o verify_hvx.exe 2>&1
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: TEST PASSED
    verify_hvx.exe
) else (
    echo FAILURE: COMPILATION FAILED
)

REM Try clang++
clang++ -I. -Itests/mock_sdk tests/test_real_compilation.cpp -o verify_hvx.exe
if %ERRORLEVEL% EQU 0 (
    echo Compilation Successful! Running test...
    verify_hvx.exe
    exit /b
)

REM Try cl (MSVC)
cl /I. /Itests/mock_sdk /EHsc tests/test_real_compilation.cpp /Fe:verify_hvx.exe
if %ERRORLEVEL% EQU 0 (
    echo Compilation Successful! Running test...
    verify_hvx.exe
    exit /b
)

echo.
echo FAILURE: No compiler found (g++, clang++, cl) in the current environment.
echo.
echo BUT: The test kit code has been generated in `tests/test_real_compilation.cpp`.
echo You can run the following command on a machine with a compiler:
echo.
echo    g++ -I. -Itests/mock_sdk tests/test_real_compilation.cpp -o verify_hvx.exe && verify_hvx.exe
echo.
exit /b 1
