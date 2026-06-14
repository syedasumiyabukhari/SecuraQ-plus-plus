@echo off
echo Starting SecuraQ++ Scanning Backend (port 8000)...
cd /d "%~dp0scanning_backend"
python backend_api.py
