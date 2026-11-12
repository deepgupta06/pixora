@echo off
rem Starts Pixora as a desktop window (Tauri shell in development mode).
rem The shell starts the local service itself and stops it when the window closes.
setlocal
title Pixora launcher

rem Where the code is: the folder in PIXORA_ROOT when that is set, else next to
rem this file, else a sibling folder (another git worktree) that holds the app.
set "ROOT=%~dp0"
if defined PIXORA_ROOT set "ROOT=%PIXORA_ROOT%\"
if not exist "%ROOT%app\src-tauri\tauri.conf.json" (
  for /d %%D in ("%~dp0..\*") do if exist "%%D\app\src-tauri\tauri.conf.json" set "ROOT=%%D\"
)
if not exist "%ROOT%app\src-tauri\tauri.conf.json" (
  echo Could not find the Pixora app folder. Put this file next to the "app" and
  echo "service" folders, or set PIXORA_ROOT to the folder that holds them.
  pause
  exit /b 1
)
cd /d "%ROOT%app"

where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js was not found. Install it from https://nodejs.org and try again.
  pause
  exit /b 1
)

where cargo >nul 2>nul
if errorlevel 1 (
  echo The Rust toolchain was not found. Install it from https://rustup.rs and try again.
  pause
  exit /b 1
)

if not exist "%ROOT%service\.venv\Scripts\python.exe" (
  echo The service environment is missing. In the "service" folder run:
  echo   py -3.12 -m venv .venv
  echo   .venv\Scripts\python -m pip install -e ".[dev]"
  pause
  exit /b 1
)

rem Use the library and models kept in the repository folder when they exist.
rem Without these the service falls back to %LOCALAPPDATA%\Pixora.
if exist "%ROOT%service\.data" set "PIXORA_DATA_DIR=%ROOT%service\.data"
if exist "%ROOT%service\.models" set "PIXORA_MODELS_DIR=%ROOT%service\.models"

if not exist node_modules (
  echo Installing app dependencies...
  call npm install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)

echo Starting Pixora. The first start compiles the shell and can take a few minutes.
echo Close the Pixora window to stop.
call npm run tauri dev
if errorlevel 1 (
  echo.
  echo Pixora stopped with an error. If the message above mentions port 5173,
  echo another copy of the app or its dev server is already running.
  pause
)
endlocal
