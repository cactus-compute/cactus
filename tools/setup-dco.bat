@echo off
setlocal enabledelayedexpansion

echo Setting up DCO for the Cactus project...

git config core.hooksPath .githooks

echo ✓ Git hooks configured to use .githooks directory

for /f "tokens=*" %%a in ('git config user.name') do set "name=%%a"
for /f "tokens=*" %%b in ('git config user.email') do set "email=%%b"

if "!name!"=="" (
    set missing=1
) else if "!email!"=="" (
    set missing=1
) else (
    set missing=0
)

if !missing!==1 (
    echo.
    echo ⚠️  Warning: Git user configuration is incomplete
    echo.
    echo Please configure your git identity:
    echo   git config --global user.name "Your Name"
    echo   git config --global user.email "your.email@example.com"
    echo.
) else (
    echo ✓ Git user configured as: !name! ^<!email!^>
)

echo.
echo DCO setup complete!
echo.
echo From now on, your commits will automatically be signed-off.
echo You can also manually sign commits with: git commit -s
echo.
echo To learn more about contributing, see CONTRIBUTING.md
