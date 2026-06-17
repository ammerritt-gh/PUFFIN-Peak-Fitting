@echo off
setlocal
title PUFFIN (dev)
cd /d "%~dp0"

REM Developer launcher: builds/uses a local .venv from the pinned lock file and runs PUFFIN.
REM Uses uv from PATH if available, otherwise the copy installed by WINDOWS-install-PUFFIN.bat.

set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 set "UV=%LOCALAPPDATA%\PUFFIN\uv\uv.exe"

if exist ".venv\Scripts\python.exe" goto run

if not exist "%UV%" (
    echo [ERROR] uv was not found on PATH or at "%LOCALAPPDATA%\PUFFIN\uv\uv.exe".
    echo         Install uv from https://docs.astral.sh/uv/ or run installer\WINDOWS-install-PUFFIN.bat.
    pause
    exit /b 1
)

echo [INFO] Creating .venv with Python 3.12 ...
"%UV%" venv .venv --python 3.12
if errorlevel 1 (
    echo [ERROR] Failed to create .venv.
    pause
    exit /b 1
)

echo [INFO] Installing pinned dependencies ...
"%UV%" pip install --python ".venv\Scripts\python.exe" -r requirements.lock.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
endlocal
