@echo off
rem ===================================================================
rem  NEUQ-VisionCalib 智能车视觉标定工具启动脚本
rem
rem  双击此文件启动本地 Web 工具。
rem  按以下顺序查找 Python 解释器：
rem    1. .venv\Scripts\python.exe   工程本地虚拟环境（推荐）
rem    2. "py"                       Windows Python 启动器
rem    3. "python"                   PATH 中的 Python
rem  如果所选解释器缺少依赖，会从 requirements.txt 安装。
rem ===================================================================

chcp 65001 >nul 2>nul
cd /d "%~dp0"
title NEUQ-VisionCalib - 智能车视觉标定工具

rem server.py 输出中文；强制使用 UTF-8，避免乱码
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
echo   [错误] 未找到可用的 Python 解释器。
echo.
echo   已检查：
echo     1. .venv\Scripts\python.exe
echo     2. py
echo     3. python
echo.
echo   请从 https://www.python.org/downloads/ 安装 Python 3.10 或更高版本，
echo   安装时勾选“将 python.exe 添加到 PATH”，然后重新运行此脚本。
echo   也可以创建工程本地虚拟环境：
echo       python -m venv .venv ^&^& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
pause
exit /b 1

:run
"%PY%" -c "import cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo.
    echo   缺少 opencv-python / numpy，正在从 requirements.txt 安装……
    echo.
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo   [错误] 依赖安装失败。请手动运行：
        echo       "%PY%" -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo.
echo   Python 解释器：%PY%
echo   工程目录：%~dp0
echo.
echo   建议点击网页右上角的“退出”按钮正常关闭，也可以直接关闭此窗口。
echo   也可以按 Ctrl+C；如果 cmd.exe 询问是否终止批处理（Y/N），请输入 Y。
echo   dist\NEUQ-VisionCalib\ 中的便携版 exe 不会出现这个询问。
echo.
"%PY%" src\webui\server.py %*

echo.
echo   本地 Web 工具已停止。
pause
