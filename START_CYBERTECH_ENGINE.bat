@echo off
REM The Cybertech Engine - starts the local OpenAI-compatible server.
REM Usage: double-click, or run:  START_CYBERTECH_ENGINE.bat qwen2.5:7b
REM The model is downloaded the first time (like "ollama pull").
setlocal
cd /d "%~dp0"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=qwen2.5:3b"
set "PORT=8080"
if not exist "cybertech-engine.exe" (
  echo cybertech-engine.exe must be in the same folder as this file.
  pause & exit /b 1
)
cybertech-engine.exe list | findstr /i /c:"%MODEL::=-%" >nul
if errorlevel 1 (
  echo Model "%MODEL%" not downloaded yet. Downloading...
  cybertech-engine.exe pull %MODEL%
  if errorlevel 1 ( echo Download failed. & pause & exit /b 1 )
)
echo Starting The Cybertech Engine on http://127.0.0.1:%PORT%  (model: %MODEL%)
echo In Cybertech AI: Settings - Answer model - "The Cybertech Engine", model name: %MODEL%
cybertech-engine.exe serve %MODEL% --host 127.0.0.1 --port %PORT%
pause
