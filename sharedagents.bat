@echo off
setlocal EnableExtensions DisableDelayedExpansion
if not "%OS%"=="Windows_NT" exit /b 2
set "SHAREDAGENTS_LAUNCH_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%SHAREDAGENTS_LAUNCH_PY%" goto missing_venv
rem UTF-8 mode: Python on Windows otherwise reads and writes text files in the
rem ANSI code page, which sharedagents's UTF-8 readers cannot decode.
set "PYTHONUTF8=1"
if not defined SHAREDAGENTS_HOME set "SHAREDAGENTS_HOME=%USERPROFILE%\.sharedagents"
if not exist "%SHAREDAGENTS_HOME%\logs\" mkdir "%SHAREDAGENTS_HOME%\logs"
if not exist "%SHAREDAGENTS_HOME%\logs\" exit /b 1
rem The helper returns after its bounded readiness checks, never for the server
rem lifetimes, and gives EACH server its own console and log handles.
rem Measured on the Windows host: `start "t" /min /wait prog >log 2>&1` leaves
rem log at 0 bytes, hiding every START/SKIP/FAIL line. Run the helper in this
rem console instead and capture its output directly; the servers still detach
rem themselves (own console, breakaway from the session job).
"%SHAREDAGENTS_LAUNCH_PY%" -m sharedagents.serve --detach >"%SHAREDAGENTS_HOME%\logs\windows-launcher.log" 2>&1
set "SHAREDAGENTS_LAUNCH_RC=%ERRORLEVEL%"
type "%SHAREDAGENTS_HOME%\logs\windows-launcher.log"
exit /b %SHAREDAGENTS_LAUNCH_RC%

:missing_venv
echo ERROR: Missing repo venv: "%SHAREDAGENTS_LAUNCH_PY%" 1>&2
echo Create it with Python 3.11.9: py -3.11 -m venv "%~dp0.venv" 1>&2
echo Then run that venv's python -m pip install -e ".[web]" from this repo. 1>&2
exit /b 2
