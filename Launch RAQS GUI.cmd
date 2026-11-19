@echo off
setlocal
pushd "%~dp0" || exit /b 1

if exist ".venv\Scripts\python.exe" goto check_environment

set "RAQS_PY=py -3.11"
py -3.11 -V >nul 2>&1
if errorlevel 1 set "RAQS_PY=python"
%RAQS_PY% -c "import struct, sys; sys.exit(sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8)" >nul 2>&1
if errorlevel 1 (
    echo Install 64-bit Python 3.11 with the Python launcher for Windows, then double-click this file again.
    echo Download: https://www.python.org/downloads/release/python-3119/
    goto setup_failed
)
echo Creating the RAQS environment...
%RAQS_PY% -m venv .venv
if errorlevel 1 goto setup_failed

:check_environment
".venv\Scripts\python.exe" -c "import struct, sys; sys.exit(sys.version_info[:2] != (3, 11) or struct.calcsize('P') != 8)"
if errorlevel 1 (
    echo RAQS needs a working 64-bit Python 3.11 environment. Rename the .venv folder and try again.
    goto setup_failed
)

echo Checking and installing RAQS dependencies. First setup can take a few minutes...
".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m ensurepip
    if errorlevel 1 goto setup_failed
)
".venv\Scripts\python.exe" -m pip --disable-pip-version-check install --quiet -r requirements-gui.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and the error above, then try again.
    goto setup_failed
)

".venv\Scripts\python.exe" -m raqs_gui
set "RAQS_EXIT=%ERRORLEVEL%"
if not "%RAQS_EXIT%"=="0" (
    echo RAQS exited with code %RAQS_EXIT%. See docs\gui.md for setup instructions.
    pause
)
popd
exit /b %RAQS_EXIT%

:setup_failed
echo See docs\gui.md for setup instructions.
pause
popd
exit /b 1
