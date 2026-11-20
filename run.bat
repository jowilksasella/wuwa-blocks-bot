@echo off
rem Pure ASCII + CRLF launcher (foreground mode; F8 to stop)
net session >nul 2>&1
if errorlevel 1 (
  if not "%~1"=="__elevated__" (
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '__elevated__' -Verb RunAs"
    exit /b
  )
)
if "%~1"=="__elevated__" shift
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
if exist "C:\ProgramData\miniconda3\python.exe" set PY="C:\ProgramData\miniconda3\python.exe"
%PY% bot.py %1 %2 %3 %4 %5 %6 %7 %8 %9
pause
