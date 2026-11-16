@echo off
chcp 65001 >nul
title Universal Model Abliterator & Q4_K_M Studio
echo ==============================================================================
echo  🚀 UNIVERSAL MODEL ABLITERATOR & Q4_K_M STUDIO
echo ==============================================================================
echo.
python run_universal_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Pipeline encountered an error.
    pause
    exit /b %errorlevel%
)
echo.
echo 🎉 Done! Model transformed and saved.
pause
