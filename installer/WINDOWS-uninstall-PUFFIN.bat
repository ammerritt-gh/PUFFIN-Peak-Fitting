@echo off
setlocal DisableDelayedExpansion

:: ============================================================================
:: PUFFIN safe uninstaller.
:: Removes only the user PUFFIN install, its bundled uv, and the shortcut.
:: ============================================================================

set "INSTALL_DIR=%USERPROFILE%\PUFFIN"
set "UV_HOME=%LOCALAPPDATA%\PUFFIN\uv"
set "SHORTCUT=%USERPROFILE%\Desktop\PUFFIN.lnk"

title PUFFIN Safe Uninstaller

echo ============================================================================
echo                          PUFFIN Safe Uninstaller
echo ============================================================================
echo.
echo This will remove:
echo   PUFFIN folder    : %INSTALL_DIR%
echo   Bundled uv       : %UV_HOME%
echo   Desktop shortcut : %SHORTCUT%
echo.
echo This will NOT remove:
echo   A system-wide uv installation (only the copy under %LOCALAPPDATA%\PUFFIN)
echo   Python or any other application
echo.
choice /C YN /M "Proceed"
if errorlevel 2 exit /b 0

echo.
echo [Step 1/3] Removing PUFFIN folder...
if not exist "%INSTALL_DIR%" goto skip_dir
if exist "%INSTALL_DIR%\main.py" goto remove_dir
echo [WARN] %INSTALL_DIR% does not contain main.py.
choice /C YN /M "Delete this folder anyway"
if errorlevel 2 goto skip_dir
:remove_dir
rd /s /q "%INSTALL_DIR%"
if errorlevel 1 (
    echo [WARN] Could not fully remove %INSTALL_DIR%. Close anything using it and retry.
) else (
    echo [OK] Removed %INSTALL_DIR%.
)
goto after_dir
:skip_dir
echo [INFO] Skipped PUFFIN folder.
:after_dir

echo.
echo [Step 2/3] Removing bundled uv...
if exist "%UV_HOME%" (
    rd /s /q "%UV_HOME%"
    echo [OK] Removed %UV_HOME%.
) else (
    echo [INFO] No bundled uv found.
)

echo.
echo [Step 3/3] Removing desktop shortcut...
if exist "%SHORTCUT%" del "%SHORTCUT%" >nul 2>nul
echo [OK] Done.
echo.
echo Note: uv may have downloaded Python under %LOCALAPPDATA%\uv (shared with any
echo other uv tools). Remove that folder manually only if you use no other uv tools.
pause
endlocal
