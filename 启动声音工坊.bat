@echo off
chcp 65001 >nul
title Voice Studio
cd /d "%~dp0GPT-SoVITS"
echo Starting Voice Studio... the browser will open http://127.0.0.1:7866
echo Keep this window open while using it. Close it to exit.
runtime\python.exe -s "%~dp0app\voice_studio.py"
pause
