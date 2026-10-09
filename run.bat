@echo off
title WatchWise Server
echo ================================
echo   WatchWise - Starting Server
echo ================================
echo.

cd /d "%~dp0"

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo.
echo Starting Django development server at http://127.0.0.1:8000
echo Press CTRL+C to stop.
echo.
start "" "http://127.0.0.1:8000"
python manage.py runserver

pause
