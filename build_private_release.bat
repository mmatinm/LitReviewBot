@echo off
setlocal EnableExtensions

cd /d "%~dp0"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%T"
set "RELEASE_DIR=release\LitReviewBot-Windows-%STAMP%"
set "ZIP_OUT=%RELEASE_DIR%.zip"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Create it first and install dependencies.
  echo Example:
  echo   py -3 -m venv venv
  echo   venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

set "VENV_PY=venv\Scripts\python.exe"

echo [1/5] Installing build tools...
"%VENV_PY%" -m pip install --upgrade pip pyinstaller
if errorlevel 1 (
  echo [ERROR] Failed to install pyinstaller.
  pause
  exit /b 1
)

echo [2/5] Cleaning previous build artifacts...
taskkill /IM LitReviewBot.exe /F >nul 2>&1
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist LitReviewBot.spec del /f /q LitReviewBot.spec >nul 2>&1

echo [3/5] Building single-file executable...
"%VENV_PY%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --name LitReviewBot ^
  --collect-all streamlit ^
  --hidden-import fitz ^
  --hidden-import app ^
  --hidden-import api_client ^
  --hidden-import config ^
  --hidden-import pdf_processor ^
  --hidden-import vector_store ^
  --hidden-import langchain_community.vectorstores.faiss ^
  --add-data "app.py;." ^
  run_bot.py

if errorlevel 1 (
  echo [ERROR] PyInstaller build failed.
  pause
  exit /b 1
)

echo [4/5] Preparing end-user package...
if not exist release mkdir release
mkdir "%RELEASE_DIR%"
copy /Y dist\LitReviewBot.exe "%RELEASE_DIR%\LitReviewBot.exe" >nul

(
  echo @echo off
  echo setlocal
  echo cd /d "%%~dp0"
  echo .\LitReviewBot.exe
) > "%RELEASE_DIR%\Run-LitReviewBot.bat"

(
  echo LITERATURE REVIEW BOT - WINDOWS PACKAGE
  echo.
  echo 1. Double-click Run-LitReviewBot.bat
  echo 2. Enter OpenRouter API key inside the app sidebar
  echo 3. Upload papers and use the bot
  echo.
  echo This package contains executable files only. No source files are shipped in the zip.
) > "%RELEASE_DIR%\README.txt"

echo [5/5] Creating zip archive...
powershell -NoProfile -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath '%ZIP_OUT%'"
if errorlevel 1 (
  echo [ERROR] Failed to create zip archive.
  pause
  exit /b 1
)

echo Done.
echo Output folder: %RELEASE_DIR%
echo Output zip   : %ZIP_OUT%
