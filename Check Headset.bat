@echo off
rem Double-click to see whether the g.tec Unicorn headset is streaming and every electrode has contact.
rem Shows one line a second: good / noisy / flat per electrode. Close the window (or Ctrl+C) to stop.
title BrainType - headset check
cd /d "%~dp0"
"venv\Scripts\python.exe" acquisition\unicorn_check.py %*
echo.
pause
