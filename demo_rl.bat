@echo off
rem Live demo of the pure RL policy that learned bank shots (strict one-contact rules, 1-3 blocks:
rem 69.5% mean on the strict test suite, 85.5% when the straight line is blocked).
rem Previous model (original rules, 1 block: 84.5% test): demo_rl.bat models\tacc-2026-10\W1-seed2
rem Press M to switch to the hand-coded strike controller and back.
rem Other models: demo_rl.bat models\tacc-2026-10\WB-seed5   or   models\tacc-2026-10\imitation-ppo-run2-12M  (79.5%, trained on a narrower block area)
cd /d "%~dp0"
set "RL_RUN=%~1"
if "%RL_RUN%"=="" set "RL_RUN=models\tacc-2026-10\AB-seed3-bank"
set "DEMO_PYTHON=python"
if exist "%USERPROFILE%\Anaconda3\python.exe" set "DEMO_PYTHON=%USERPROFILE%\Anaconda3\python.exe"
"%DEMO_PYTHON%" -m scripts.demo_live --rl-run "%RL_RUN%" --start-rl
if errorlevel 1 pause
