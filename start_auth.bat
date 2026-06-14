@echo off
echo Starting SecuraQ++ Auth Backend (port 4000)...
cd /d "%~dp0auth_backend"
npm install --silent
node server.js
