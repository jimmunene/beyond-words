@echo off
rem Double-click to run BrainType LIVE on the g.tec Unicorn headset.
rem First: in Unicorn Suite open "Unicorn LSL", connect the headset and press Start.
rem Simulation instead (recorded EEG, no headset): double-click "Start BrainType.bat".
title BrainType - live headset
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo The Python environment "venv" was not found in this folder.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" speller\launch.py --source headset %*
if errorlevel 1 (
    echo.
    echo BrainType did not start properly. Read the message above.
    pause
)
