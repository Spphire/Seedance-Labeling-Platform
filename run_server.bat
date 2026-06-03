@echo off
setlocal
cd /d "%~dp0"
if "%SEEDANCE_HOST%"=="" set "SEEDANCE_HOST=0.0.0.0"
if "%SEEDANCE_PORT%"=="" set "SEEDANCE_PORT=18080"
set "SEEDANCE_RELOAD_ARG="
if "%SEEDANCE_RELOAD%"=="1" set "SEEDANCE_RELOAD_ARG=--reload"
".venv\Scripts\python.exe" -m uvicorn app.backend.main:app --host "%SEEDANCE_HOST%" --port "%SEEDANCE_PORT%" %SEEDANCE_RELOAD_ARG%
