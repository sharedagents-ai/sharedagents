@echo off
setlocal EnableExtensions DisableDelayedExpansion
if not "%OS%"=="Windows_NT" exit /b 2
set "SHAREDAGENTS_LAUNCH_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%SHAREDAGENTS_LAUNCH_PY%" goto missing_venv
rem UTF-8 mode: Python on Windows otherwise reads and writes text files in the
rem ANSI code page, which sharedagents's UTF-8 readers cannot decode.
set "PYTHONUTF8=1"
"%SHAREDAGENTS_LAUNCH_PY%" -m sharedagents.serve status
exit /b %ERRORLEVEL%

:missing_venv
echo ERROR: Missing repo venv: "%SHAREDAGENTS_LAUNCH_PY%" 1>&2
echo Create it with Python 3.11.9: py -3.11 -m venv "%~dp0.venv" 1>&2
echo Then run that venv's python -m pip install -e ".[web]" from this repo. 1>&2
exit /b 2
