@echo off
rem Pure ASCII + CRLF launcher (dry run, saves preview.png)
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
if exist "C:\ProgramData\miniconda3\python.exe" set PY="C:\ProgramData\miniconda3\python.exe"
%PY% bot.py --dry %*
if exist preview.png start "" preview.png
pause
