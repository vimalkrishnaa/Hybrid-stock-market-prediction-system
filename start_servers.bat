@echo off
echo ============================================
echo   IndiTrendAI - Starting Both Servers
echo ============================================
echo.
echo Starting Backend Server (Port 8000)...
start "IndiTrendAI Backend" cmd /k "cd /d %~dp0 && python api_server.py"
timeout /t 3 /nobreak >nul
echo.
echo Starting Frontend Server (Port 3000)...
start "IndiTrendAI Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"
echo.
echo ============================================
echo   Servers are starting in separate windows
echo ============================================
echo.
echo Backend API:  http://localhost:8000
echo Frontend:     http://localhost:3000
echo API Docs:     http://localhost:8000/docs
echo.
echo Press any key to exit this window...
pause >nul

