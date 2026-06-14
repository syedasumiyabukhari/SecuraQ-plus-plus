@echo off
echo Starting SecuraQ++ Frontend (port 5173)...
cd /d "%~dp0frontend"
npm install --silent
npm run dev
