@echo off
rem Double-click to start the whole BrainType demo: recorded g.tec EEG + classifier + speller.
rem Other choices:  "Start BrainType.bat" --dataset both     "Start BrainType.bat" --source keyboard
title BrainType
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo The Python environment "venv" was not found in this folder.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" speller\launch.py %*
if errorlevel 1 (
    echo.
    echo BrainType did not start properly. Read the message above.
    pause
)
