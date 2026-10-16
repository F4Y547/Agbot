# Sets up the local agent environment (Windows / PowerShell).
#   powershell -ExecutionPolicy Bypass -File dev\setup.ps1
param(
    [string]$OllamaModel = "qwen3:8b"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> Creating venv (.venv)"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -q -U pip
& .\.venv\Scripts\python.exe -m pip install -q -r requirements.txt
Write-Host "==> Python deps installed"

# Ollama fallback model
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $models = (ollama list) -join "`n"
    if ($models -match [regex]::Escape($OllamaModel)) {
        Write-Host "==> Ollama model '$OllamaModel' already present"
    } else {
        Write-Host "==> Pulling Ollama model '$OllamaModel' (~5 GB, one-time)"
        ollama pull $OllamaModel
    }
} else {
    Write-Host "!! Ollama not found - install it from https://ollama.com/download for the local fallback"
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "==> Created .env from .env.example (edit AGENT_BASE_URL when your GPU box is up)"
}

Write-Host ""
Write-Host "Done. Try:"
Write-Host "  .\.venv\Scripts\python.exe -m agbot --selftest"
Write-Host "  .\.venv\Scripts\python.exe -m agbot"
