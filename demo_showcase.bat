@echo off
rem Live demo of the RL policy on pre-checked shots it scores (N cycles through them).
rem Clicking to place obstacles still plays unselected shots. Press M for the strike controller.
cd /d "%~dp0"
set "RL_RUN=%~1"
if "%RL_RUN%"=="" set "RL_RUN=models\tacc-2026-10\AB-seed3-bank"
set "DEMO_PYTHON=python"
if exist "%USERPROFILE%\Anaconda3\python.exe" set "DEMO_PYTHON=%USERPROFILE%\Anaconda3\python.exe"
"%DEMO_PYTHON%" -m scripts.demo_live --rl-run "%RL_RUN%" --start-rl --showcase 40
if errorlevel 1 pause
