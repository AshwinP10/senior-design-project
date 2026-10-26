@echo off
rem Live demo of the strike controller in the real simulator. Double-click, or run from a terminal.
rem Optional: demo_live.bat --width 1600   (bigger window)   |   demo_live.bat --selftest
cd /d "%~dp0"
set "DEMO_PYTHON=python"
if exist "%USERPROFILE%\Anaconda3\python.exe" set "DEMO_PYTHON=%USERPROFILE%\Anaconda3\python.exe"
"%DEMO_PYTHON%" -m scripts.demo_live %*
if errorlevel 1 pause
