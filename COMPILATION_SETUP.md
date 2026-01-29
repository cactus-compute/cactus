# How to Install a C++ Compiler on Windows

You need a C++ compiler to run the verification test (`run_cpp_verification.bat`). Choose **Method 1** (easiest/fastest) or **Method 2** (standard).

## Method 1: Install MinGW via WinGet (The "One-Liner")
If you are on Windows 10/11, open PowerShell as Administrator and run:

```powershell
winget install -e --id MSYS2.MSYS2
```

1.  After it installs, open the **MSYS2 MSYS** terminal from your Start Menu.
2.  Run this command to install the toolchain:
    ```bash
    pacman -S --noconfirm mingw-w64-x86_64-gcc
    ```
3.  Add the compiler to your path:
    - Search "Edit the system environment variables" in Windows Start.
    - Click "Environment Variables".
    - Under "System variables", select `Path` -> Edit -> New.
    - Add: `C:\msys64\mingw64\bin` (or wherever it installed).
4.  Open a new terminal (PowerShell) and type `g++ --version` to verify.

## Method 2: Visual Studio Build Tools (The "Standard" Way)
1.  Download the **[Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)**.
2.  Run the installer.
3.  Select the **"Desktop development with C++"** workload.
4.  Click **Install** (approx. 2-3GB).
5.  Once done, search Start for **"x64 Native Tools Command Prompt for VS 2022"**.
6.  Run `cl` to verify it works.

## Running the Test
Once installed, use that terminal (where `g++` or `cl` works) to run:

```powershell
cd c:\cactus
.\run_cpp_verification.bat
```
