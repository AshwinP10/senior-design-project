@echo off
rem Live demo of the pure RL policy (DAgger + PPO; one block anywhere: 84.5% test, 91.5% validation).
rem Press M to switch to the hand-coded strike controller and back.
rem Other models: demo_rl.bat models\tacc-2026-10\WB-seed5   or   models\tacc-2026-10\imitation-ppo-run2-12M  (79.5%, trained on a narrower block area)
cd /d "%~dp0"
set "RL_RUN=%~1"
if "%RL_RUN%"=="" set "RL_RUN=models\tacc-2026-10\W1-seed2"
set "DEMO_PYTHON=python"
if exist "%USERPROFILE%\Anaconda3\python.exe" set "DEMO_PYTHON=%USERPROFILE%\Anaconda3\python.exe"
"%DEMO_PYTHON%" -m scripts.demo_live --rl-run "%RL_RUN%" --start-rl
if errorlevel 1 pause
