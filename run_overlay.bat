@echo off
rem Pure ASCII + CRLF launcher (click-through overlay, no mouse input)
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
if exist "C:\ProgramData\miniconda3\python.exe" set PY="C:\ProgramData\miniconda3\python.exe"
%PY% overlay.py %*
pause
