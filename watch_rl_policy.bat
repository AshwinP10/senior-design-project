@echo off
cd /d "%~dp0"
set "POLICY_RUN=%~1"
if "%POLICY_RUN%"=="" set "POLICY_RUN=runs\incoming-v2-final"
if not exist "%POLICY_RUN%\policy.zip" if "%~1"=="" set "POLICY_RUN=models\precision-striker-v2"
set "POLICY_PYTHON=python"
if exist "%USERPROFILE%\Anaconda3\python.exe" set "POLICY_PYTHON=%USERPROFILE%\Anaconda3\python.exe"
"%POLICY_PYTHON%" -m scripts.watch_precision_striker --run "%POLICY_RUN%"
if errorlevel 1 pause
