@echo off
title WatchWise Server
echo ================================
echo   WatchWise - Starting Server
echo ================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\Scripts\python.exe
    echo Create it first:
    echo   uv venv .venv --python 3.11
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Applying migrations...
".venv\Scripts\python.exe" manage.py migrate
if errorlevel 1 (
    echo ERROR: Migrations failed.
    pause
    exit /b 1
)

echo.
echo Starting Django development server at http://127.0.0.1:8000
echo Press CTRL+C to stop.
echo.
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" manage.py runserver 127.0.0.1:8000

pause
