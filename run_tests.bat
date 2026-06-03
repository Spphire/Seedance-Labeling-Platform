@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m unittest discover -s tests -p "test_*.py"
