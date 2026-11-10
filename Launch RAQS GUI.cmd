@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo RAQS could not find .venv\Scripts\python.exe.
    echo See docs\gui.md for setup instructions.
    pause
    popd
    exit /b 1
)

".venv\Scripts\python.exe" -m raqs_gui
set "RAQS_EXIT=%ERRORLEVEL%"
if not "%RAQS_EXIT%"=="0" (
    echo RAQS exited with code %RAQS_EXIT%. See docs\gui.md for setup instructions.
    pause
)
popd
exit /b %RAQS_EXIT%
