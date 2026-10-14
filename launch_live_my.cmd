@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0native_runner\launch_live_my.ps1" %*
exit /b %ERRORLEVEL%
