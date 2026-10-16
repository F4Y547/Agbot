@echo off
rem Run Agbot from the repo root:  run.cmd [prompt] [options]
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found - set it up first:
  echo   powershell -ExecutionPolicy Bypass -File dev\setup.ps1
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m agbot %*
