@echo off
echo Building Cube2ICC.exe...

REM Ensure uv is installed or python environment is set up
REM We try to run with uv directly if available
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run pyinstaller --noconfirm --clean --name "Cube2ICC" --windowed gui.py
) else (
    echo "uv not found, trying direct python/pyinstaller..."
    pyinstaller --noconfirm --clean --name "Cube2ICC" --windowed gui.py
)

if %errorlevel%==0 (
    echo.
    echo -------------------------------------------------------
    echo Build complete. App is located in dist\Cube2ICC\Cube2ICC.exe
    echo -------------------------------------------------------
) else (
    echo.
    echo -------------------------------------------------------
    echo Build FAILED. Please check the logs above.
    echo Ensure you have pyinstaller installed: pip install pyinstaller
    echo -------------------------------------------------------
    pause
    exit /b 1
)

pause
