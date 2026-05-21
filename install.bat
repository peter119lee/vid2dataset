@echo off
chcp 65001 >nul
title vid2dataset Installer
echo.
echo  ╔══════════════════════════════════════╗
echo  ║       vid2dataset - Installer        ║
echo  ╚══════════════════════════════════════╝
echo.

:: Check if Python exists
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python not found. Installing Python via winget...
    echo.
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install Python.
        echo Please install Python 3.10+ manually from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Python installed. You may need to restart this script.
    echo.
)

echo [1/3] Creating virtual environment...
python -m venv "%~dp0.venv"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)

echo [2/3] Installing vid2dataset...
"%~dp0.venv\Scripts\pip.exe" install -e "%~dp0" --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install. Check your internet connection.
    pause
    exit /b 1
)

echo [3/3] Creating desktop shortcut...
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $sc = $ws.CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'vid2dataset.lnk')); ^
   $sc.TargetPath = '%~dp0.venv\Scripts\pythonw.exe'; ^
   $sc.Arguments = '-m vid2dataset app'; ^
   $sc.WorkingDirectory = '%~dp0'; ^
   $sc.IconLocation = '%~dp0.venv\Scripts\python.exe,0'; ^
   $sc.Description = 'vid2dataset - Video to Training Set'; ^
   $sc.Save()"

echo.
echo  ╔══════════════════════════════════════╗
echo  ║          Install Complete!           ║
echo  ╠══════════════════════════════════════╣
echo  ║  A shortcut has been created on     ║
echo  ║  your Desktop. Double-click it      ║
echo  ║  to launch vid2dataset.             ║
echo  ╚══════════════════════════════════════╝
echo.
pause
