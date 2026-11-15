@echo off
rem Double-click to test BrainType with the keyboard: no EEG, keys 1-4 (or mouse clicks) pick the boxes.
title BrainType - keyboard test
cd /d "%~dp0"
"venv\Scripts\python.exe" speller\launch.py --source keyboard %*
if errorlevel 1 (
    echo.
    echo BrainType did not start properly. Read the message above.
    pause
)
