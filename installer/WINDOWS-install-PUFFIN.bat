@echo off
setlocal DisableDelayedExpansion

:: ============================================================================
:: PUFFIN Windows installer - release-pinned
:: Pure-Python install via uv. No administrator rights required.
:: Installs only under the current user's profile.
:: ============================================================================

set "PUFFIN_VERSION=v0.3.0-alpha"
set "INSTALLER_VERSION=v0.3.0-alpha"
set "PYTHON_VERSION=3.12"
set "UV_VERSION=0.11.18"
set "UV_SHA256=bf8e0021336b7c77bd80a078b612125f385b08f541437edaea8c8ca9e574db0d"
set "REPO_SLUG=ammerritt-gh/PUFFIN-Peak-Fitting"

set "INSTALL_DIR=%USERPROFILE%\PUFFIN"
set "UV_HOME=%LOCALAPPDATA%\PUFFIN\uv"
set "UV_EXE=%UV_HOME%\uv.exe"
set "SHORTCUT=%USERPROFILE%\Desktop\PUFFIN.lnk"
set "ZIP_URL=https://github.com/%REPO_SLUG%/archive/refs/tags/%PUFFIN_VERSION%.zip"
set "UV_ZIP_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"

title PUFFIN Installer

echo ============================================================================
echo                         PUFFIN Installation Script
echo                        Peak Fitting for 1D Spectra
echo                          Release: %PUFFIN_VERSION%
echo ============================================================================
echo Installer version: %INSTALLER_VERSION%
echo PUFFIN version:    %PUFFIN_VERSION%
echo Python version:    %PYTHON_VERSION% (provided privately by uv)
echo uv version:        %UV_VERSION%
echo.
echo This installer will set up PUFFIN for this Windows user account. It will:
echo   1. Download uv (a small, self-contained tool) to:
echo      %UV_HOME%
echo   2. Install a private copy of Python %PYTHON_VERSION% via uv.
echo   3. Download PUFFIN %PUFFIN_VERSION% to:
echo      %INSTALL_DIR%
echo   4. Create a virtual environment and install pinned dependencies.
echo   5. Create a desktop shortcut named "PUFFIN".
echo.
echo It does NOT require administrator rights, does NOT modify the system
echo Python or PATH, and installs only under your user profile.
echo.
choice /C YN /M "Continue with PUFFIN installation"
if errorlevel 2 (
    echo Installation cancelled.
    pause
    exit /b 0
)
echo.

echo [Step 1/5] Setting up uv...
if not exist "%UV_HOME%" mkdir "%UV_HOME%"
if exist "%UV_EXE%" goto uv_ready
echo [INFO] Downloading uv %UV_VERSION%...
set "UV_ZIP=%TEMP%\puffin_uv_%RANDOM%.zip"
curl -L -o "%UV_ZIP%" "%UV_ZIP_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to download uv.
    pause
    exit /b 1
)
echo [INFO] Verifying download checksum...
set "UV_HASH="
for /f "usebackq delims=" %%H in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%UV_ZIP%').Hash.ToLower()"`) do set "UV_HASH=%%H"
if /i not "%UV_HASH%"=="%UV_SHA256%" (
    echo [ERROR] uv checksum mismatch - aborting.
    echo         Expected: %UV_SHA256%
    echo         Actual:   %UV_HASH%
    del "%UV_ZIP%" >nul 2>nul
    pause
    exit /b 1
)
echo [INFO] Extracting uv...
tar -xf "%UV_ZIP%" -C "%UV_HOME%"
if errorlevel 1 (
    echo [ERROR] Failed to extract uv.
    del "%UV_ZIP%" >nul 2>nul
    pause
    exit /b 1
)
del "%UV_ZIP%" >nul 2>nul
:uv_ready
if not exist "%UV_EXE%" (
    echo [ERROR] uv.exe not found at %UV_EXE%
    pause
    exit /b 1
)
"%UV_EXE%" --version
echo [OK] uv ready.
echo.

echo [Step 2/5] Installing Python %PYTHON_VERSION% via uv...
"%UV_EXE%" python install %PYTHON_VERSION%
if errorlevel 1 (
    echo [ERROR] Failed to install Python %PYTHON_VERSION%.
    pause
    exit /b 1
)
echo [OK] Python ready.
echo.

echo [Step 3/5] Downloading PUFFIN source (%PUFFIN_VERSION%)...
if exist "%INSTALL_DIR%\main.py" goto src_dir_ready
if not exist "%INSTALL_DIR%" goto src_dir_ready
echo [WARN] %INSTALL_DIR% exists but does not look like a PUFFIN install.
set "BACKUP_DIR=%USERPROFILE%\PUFFIN_backup_%RANDOM%_%RANDOM%"
echo [INFO] Moving it aside to: %BACKUP_DIR%
move "%INSTALL_DIR%" "%BACKUP_DIR%" >nul
if errorlevel 1 (
    echo [ERROR] Could not move existing folder. Close anything using it and retry.
    pause
    exit /b 1
)
:src_dir_ready
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

set "SRC_ZIP=%TEMP%\puffin_src_%RANDOM%.zip"
set "SRC_TMP=%TEMP%\puffin_src_%RANDOM%_dir"
echo [INFO] Downloading %ZIP_URL% ...
curl -L -o "%SRC_ZIP%" "%ZIP_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to download PUFFIN source.
    pause
    exit /b 1
)
mkdir "%SRC_TMP%" 2>nul
echo [INFO] Extracting...
tar -xf "%SRC_ZIP%" -C "%SRC_TMP%"
if errorlevel 1 (
    echo [ERROR] Failed to extract PUFFIN source.
    del "%SRC_ZIP%" >nul 2>nul
    pause
    exit /b 1
)
set "SRC_DIR="
for /d %%D in ("%SRC_TMP%\*") do set "SRC_DIR=%%D"
if "%SRC_DIR%"=="" (
    echo [ERROR] Could not locate extracted source folder.
    pause
    exit /b 1
)
echo [INFO] Installing files into %INSTALL_DIR% (preserving any existing .venv)...
robocopy "%SRC_DIR%" "%INSTALL_DIR%" /E /XD .venv >nul
if errorlevel 8 (
    echo [ERROR] Failed to copy PUFFIN source into %INSTALL_DIR%.
    pause
    exit /b 1
)
del "%SRC_ZIP%" >nul 2>nul
rd /s /q "%SRC_TMP%" >nul 2>nul

if not exist "%INSTALL_DIR%\main.py" (
    echo [ERROR] main.py not found after download.
    pause
    exit /b 1
)
if not exist "%INSTALL_DIR%\requirements.lock.txt" (
    echo [ERROR] requirements.lock.txt not found after download.
    echo [INFO] This release tag may predate the pinned installer files.
    pause
    exit /b 1
)
echo [OK] PUFFIN source ready.
echo.

echo [Step 4/5] Creating environment and installing dependencies...
"%UV_EXE%" venv "%INSTALL_DIR%\.venv" --python %PYTHON_VERSION%
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
"%UV_EXE%" pip install --python "%INSTALL_DIR%\.venv\Scripts\python.exe" -r "%INSTALL_DIR%\requirements.lock.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Environment ready.
echo.

echo [Step 5/5] Creating desktop shortcut...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut($env:SHORTCUT); $s.TargetPath=$env:INSTALL_DIR + '\PUFFIN-Launcher.bat'; $s.WorkingDirectory=$env:INSTALL_DIR; $s.Description='PUFFIN Launcher'; $s.Save()" 2>nul

echo PUFFIN_VERSION=%PUFFIN_VERSION%> "%INSTALL_DIR%\INSTALL_INFO.txt"
echo INSTALLER_VERSION=%INSTALLER_VERSION%>> "%INSTALL_DIR%\INSTALL_INFO.txt"
echo PYTHON_VERSION=%PYTHON_VERSION%>> "%INSTALL_DIR%\INSTALL_INFO.txt"
echo UV_VERSION=%UV_VERSION%>> "%INSTALL_DIR%\INSTALL_INFO.txt"
echo INSTALL_DIR=%INSTALL_DIR%>> "%INSTALL_DIR%\INSTALL_INFO.txt"
echo REPO_URL=https://github.com/%REPO_SLUG%>> "%INSTALL_DIR%\INSTALL_INFO.txt"

echo.
echo ============================================================================
echo Installation complete.
echo Installed to: %INSTALL_DIR%
echo PUFFIN      : %PUFFIN_VERSION%
echo Shortcut    : %SHORTCUT%
echo ============================================================================
echo.
echo Launch PUFFIN from the "PUFFIN" desktop shortcut, or run:
echo   %INSTALL_DIR%\PUFFIN-Launcher.bat
echo.
pause
endlocal
