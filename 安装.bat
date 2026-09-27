@echo off
chcp 65001 >nul
title Voice Studio - Install
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
