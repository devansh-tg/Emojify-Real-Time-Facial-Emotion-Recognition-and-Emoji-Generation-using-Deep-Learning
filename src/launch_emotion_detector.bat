@echo off
title Emotion Detection System
echo ================================================
echo   EMOTION DETECTION SYSTEM
echo   Ultra Advanced AI Emotion Analyzer
echo ================================================
echo.
echo Loading AI Model...
echo.

cd /d "%~dp0"
"C:\emojify project\EmoteVision\.venv\Scripts\python.exe" gui_advanced.py

pause
