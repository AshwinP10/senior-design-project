@echo off
cd /d "%~dp0"
echo Starting flat table sim...
echo.
python scripts/play_flat_table.py
echo.
echo Sim closed. Press any key to close this window.
pause >nul
