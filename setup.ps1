# Cactus setup script for Windows PowerShell
# Usage: . .\setup.ps1   (dot-source to activate in current session)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PythonDir = Join-Path $ScriptDir "python"
$VenvDir = Join-Path $PythonDir ".venv"

# Create venv if it doesn't exist
if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create venv. Make sure Python 3.10+ is installed." -ForegroundColor Red
        return
    }
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvScripts = Join-Path $VenvDir "Scripts"

# Install/update the cactus package in editable mode
Write-Host "Installing cactus package..." -ForegroundColor Yellow
& $VenvPython -m pip install -e $PythonDir --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: pip install failed." -ForegroundColor Red
    return
}

# Add venv Scripts to PATH for this session (so 'cactus' command works)
if ($env:PATH -notlike "*$VenvScripts*") {
    $env:PATH = "$VenvScripts;$env:PATH"
}

# Set CACTUS_PROJECT_ROOT so CLI finds the repo regardless of cwd
$env:CACTUS_PROJECT_ROOT = $ScriptDir

Write-Host "Done. 'cactus' command is now available in this session." -ForegroundColor Green
Write-Host "  cactus download openai/whisper-tiny" -ForegroundColor Cyan
Write-Host "  cactus transcribe openai/whisper-tiny --file audio.wav" -ForegroundColor Cyan
Write-Host "  cactus run LiquidAI/LFM2.5-1.2B-Instruct" -ForegroundColor Cyan
