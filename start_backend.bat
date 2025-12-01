@echo off
echo Starting IndiTrendAI Backend Server...
echo.
cd /d "%~dp0"
python api_server.py
pause

