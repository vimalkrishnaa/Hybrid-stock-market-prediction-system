@echo off
echo Starting IndiTrendAI Frontend Server...
echo.
cd /d "%~dp0\frontend"
call npm run dev
pause

