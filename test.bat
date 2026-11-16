@echo off
setlocal
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

python "%~dp0benchmark_test.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Test encountered an error.
    pause
    exit /b 1
)
pause
