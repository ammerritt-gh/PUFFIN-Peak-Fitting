@echo off
setlocal DisableDelayedExpansion
title PUFFIN Update / Repair
cd /d "%~dp0"

REM Re-downloads the pinned release source over this folder (preserving .venv) and
REM re-installs the pinned dependencies. To move to a NEWER release, download and run
REM that release's installer.

set "PUFFIN_VERSION=v0.3.0-alpha"
set "REPO_SLUG=ammerritt-gh/PUFFIN-Peak-Fitting"
set "ZIP_URL=https://github.com/%REPO_SLUG%/archive/refs/tags/%PUFFIN_VERSION%.zip"

set "UV=uv"
where uv >nul 2>nul
if errorlevel 1 set "UV=%LOCALAPPDATA%\PUFFIN\uv\uv.exe"

echo ============================================================================
echo                         PUFFIN Update / Repair
echo                         Release: %PUFFIN_VERSION%
echo ============================================================================
echo This re-downloads the pinned %PUFFIN_VERSION% source over this folder
echo (your .venv is preserved) and re-installs the pinned dependencies.
echo To move to a NEWER release, download and run that release's installer.
echo.
choice /C YN /M "Continue"
if errorlevel 2 exit /b 0

set "TMPZIP=%TEMP%\puffin_update_%RANDOM%.zip"
set "TMPDIR=%TEMP%\puffin_update_%RANDOM%"

echo [INFO] Downloading %ZIP_URL% ...
curl -L -o "%TMPZIP%" "%ZIP_URL%"
if errorlevel 1 (
    echo [ERROR] Download failed.
    pause
    exit /b 1
)

mkdir "%TMPDIR%" 2>nul
echo [INFO] Extracting ...
tar -xf "%TMPZIP%" -C "%TMPDIR%"
if errorlevel 1 (
    echo [ERROR] Extract failed.
    pause
    exit /b 1
)

set "SRCDIR="
for /d %%D in ("%TMPDIR%\*") do set "SRCDIR=%%D"
if "%SRCDIR%"=="" (
    echo [ERROR] Could not locate extracted source folder.
    pause
    exit /b 1
)

echo [INFO] Updating files (preserving .venv) ...
robocopy "%SRCDIR%" "%~dp0." /E /XD .venv >nul
if errorlevel 8 (
    echo [ERROR] Copy failed.
    pause
    exit /b 1
)

echo [INFO] Re-installing pinned dependencies ...
"%UV%" pip sync --python ".venv\Scripts\python.exe" requirements.lock.txt
if errorlevel 1 (
    echo [ERROR] Dependency sync failed.
    pause
    exit /b 1
)

del "%TMPZIP%" >nul 2>nul
rd /s /q "%TMPDIR%" >nul 2>nul

echo.
echo [OK] PUFFIN %PUFFIN_VERSION% refreshed.
pause
endlocal
