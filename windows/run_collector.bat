@echo off
REM Keeps one collector alive forever. Invoked by Task Scheduler.
REM   windows\run_collector.bat btcusdt
setlocal
set SYMBOL=%1
if "%SYMBOL%"=="" set SYMBOL=btcusdt
cd /d "%~dp0\.."
if not exist logs mkdir logs
set LOG=logs\collector_%SYMBOL%.log

:loop
echo [%date% %time%] starting %SYMBOL% >> "%LOG%"
".venv\Scripts\python.exe" -u collect.py --symbol %SYMBOL% >> "%LOG%" 2>&1
echo [%date% %time%] exited (code %errorlevel%), restarting in 10s >> "%LOG%"
REM ping instead of timeout: timeout fails without an interactive console
ping -n 11 127.0.0.1 > nul
goto loop
