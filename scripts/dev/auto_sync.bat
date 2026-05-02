@echo off
REM Launcher for auto_sync.py — keeps the window open on crash so you can
REM read the error. Adjust PYTHON if you have a venv.
setlocal

set REPO=%~dp0..\..
cd /d "%REPO%"

set PYTHON=python
%PYTHON% scripts\dev\auto_sync.py %*

if errorlevel 1 (
    echo.
    echo auto_sync.py exited with errorlevel %errorlevel%.
    echo Press any key to close...
    pause >nul
)

endlocal
