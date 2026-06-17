@echo off
setlocal
title PUFFIN
cd /d "%~dp0"

REM Installed-copy launcher. Runs PUFFIN from the bundled virtual environment.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] PUFFIN environment not found at "%~dp0.venv".
    echo         Run update-puffin.bat or re-run the PUFFIN installer.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
endlocal
