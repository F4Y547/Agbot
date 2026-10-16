# Smoke-test the agent loop against whatever endpoint is configured.
#   powershell -ExecutionPolicy Bypass -File dev\smoke.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "No .venv found - run dev\setup.ps1 first" -ForegroundColor Red
    exit 1
}

# make sure the Ollama server is up for the fallback path
if (Get-Command ollama -ErrorAction SilentlyContinue) { ollama list | Out-Null }

& .\.venv\Scripts\python.exe -m agbot --selftest
exit $LASTEXITCODE
