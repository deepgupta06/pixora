@echo off
rem Starts Pixora as a desktop window (Tauri shell in development mode).
rem The shell starts the local service itself and stops it when the window closes.
setlocal
title Pixora launcher

rem Where the code is: the folder in PIXORA_ROOT when that is set. Otherwise the
rem first folder that holds the app and its service environment, looking next to
rem this file, then in the main checkout (when this file is in a git worktree
rem under .claude\worktrees), then in the sibling worktrees. If none is ready to
rem run, the first folder with the app is used, so the checks below say what is
rem missing.
set "ROOT="
if defined PIXORA_ROOT (
  set "ROOT=%PIXORA_ROOT%\"
  goto :found
)
for %%M in (ready app) do (
  call :pick %%M "%~dp0" && goto :found
  for %%I in ("%~dp0..\..\..") do call :pick %%M "%%~fI\" && goto :found
  for /d %%D in ("%~dp0..\*") do call :pick %%M "%%~fD\" && goto :found
)
:found
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
exit /b

rem :pick ready|app FOLDER -- use FOLDER if it holds the app (and, for "ready",
rem the service environment too).
:pick
if not exist "%~2app\src-tauri\tauri.conf.json" exit /b 1
if "%~1"=="ready" if not exist "%~2service\.venv\Scripts\python.exe" exit /b 1
set "ROOT=%~2"
exit /b 0
