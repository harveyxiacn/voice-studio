@echo off
chcp 65001 >nul
title Voice Studio - Training
cd /d "%~dp0GPT-SoVITS"
runtime\python.exe -s "%~dp0app\cli.py" %*
pause
