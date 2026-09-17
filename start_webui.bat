@echo off
rem ===================================================================
rem  NEUQ vision calibration console - launcher
rem
rem  Double-click this file to start the local web console.
rem  It picks a Python interpreter in this order:
rem    1. .venv\Scripts\python.exe   project-local venv (recommended)
rem    2. "py"                       the official Windows Python launcher
rem    3. "python"                   whatever is on PATH
rem  If the chosen interpreter lacks the dependencies, it installs them
rem  from requirements.txt.
rem ===================================================================

chcp 65001 >nul 2>nul
cd /d "%~dp0"
title NEUQ vision calibration console

rem server.py prints Chinese; force UTF-8 so it renders instead of mojibake
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto run

set "PY=py"
py --version >nul 2>nul
if not errorlevel 1 goto run

set "PY=python"
python --version >nul 2>nul
if not errorlevel 1 goto run

echo.
echo   [ERROR] No usable Python interpreter found.
echo.
echo   Tried:
echo     1. .venv\Scripts\python.exe
echo     2. py
echo     3. python
echo.
echo   Install Python 3.10+ from https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" during setup, then run this again.
echo   To create a project-local venv:
echo       python -m venv .venv ^&^& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
pause
exit /b 1

:run
"%PY%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   Missing opencv-python / numpy. Installing from requirements.txt ...
    echo.
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [ERROR] Dependency install failed. Run this manually:
        echo       "%PY%" -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo.
echo   Python : %PY%
echo   Project: %~dp0
echo.
echo   Tip: to stop the server press Ctrl+C, then answer Y to the
echo        "Terminate batch job" prompt.  (The packaged exe in
echo        dist\NEUQ-VisionCalib\ does not ask - it just exits.)
echo.
"%PY%" src\webui\server.py %*

echo.
echo   Console stopped.
pause
