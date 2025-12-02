@echo off
title Emotion Detection Web App
color 0A
echo ================================================
echo   EMOTION DETECTION WEB APPLICATION
echo   Starting Flask Server...
echo ================================================
echo.

cd /d "%~dp0"

echo [1/3] Activating virtual environment...
call "C:\emojify project\emojify (copy)\.venv\Scripts\activate.bat"

echo [2/3] Installing Flask dependencies...
pip install flask werkzeug --quiet

echo [3/3] Starting web server...
echo.
echo ================================================
echo   SERVER STARTING...
echo   Open browser at: http://localhost:5000
echo   Press Ctrl+C to stop server
echo ================================================
echo.

python app.py

pause
