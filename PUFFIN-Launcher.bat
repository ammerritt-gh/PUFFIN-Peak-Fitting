@echo off
setlocal
title PUFFIN Launcher
cd /d "%~dp0"

:menu
cls
echo ============================================================================
echo                              PUFFIN Launcher
echo                    Peak Fitting for 1D Spectra - v0.3.0-alpha
echo ============================================================================
echo.
echo    [1] Run PUFFIN
echo    [2] Update / repair PUFFIN
echo    [3] Open PUFFIN folder
echo    [4] Open PUFFIN shell
echo    [5] Exit
echo.
choice /C 12345 /M "Select option"
if errorlevel 5 goto end
if errorlevel 4 goto shell
if errorlevel 3 goto folder
if errorlevel 2 goto update
if errorlevel 1 goto run
goto menu

:run
call "%~dp0run-puffin.bat"
goto menu

:update
call "%~dp0update-puffin.bat"
goto menu

:folder
explorer "%~dp0."
goto menu

:shell
cmd /k call "%~dp0.venv\Scripts\activate.bat"
goto menu

:end
endlocal
