@echo off
setlocal
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

python "%~dp0chat_cli.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Chat encountered an error.
    pause
    exit /b 1
)
pause
