@echo off
REM Windows Setup Script for InspireFace WSL Environment
REM Run this as Administrator

echo 🚀 Setting up WSL for InspireFace
echo ==================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running as Administrator
) else (
    echo ❌ Please run as Administrator!
    pause
    exit /b 1
)

echo 📦 Installing WSL...
wsl --install -d Ubuntu

echo ⏳ WSL installation started. Please wait...
echo 📝 After installation completes:
echo    1. Open Ubuntu from Start Menu
echo    2. Run: sudo apt update && sudo apt upgrade -y
echo    3. Copy setup_wsl_inspireface.sh to WSL
echo    4. Run: bash setup_wsl_inspireface.sh
echo    5. Follow INSPIREFACE_INTEGRATION_GUIDE.md

echo.
echo 🎯 What happens next:
echo =====================
echo 1. WSL Ubuntu will be installed
echo 2. Restart your computer if prompted
echo 3. Open Ubuntu terminal
echo 4. Copy project files to WSL:
echo    cp -r /mnt/c/Users/%USERNAME%/Documents/GitHub/gender_detect ~
echo 5. Run the WSL setup script
echo.
echo 📚 Documentation: INSPIREFACE_INTEGRATION_GUIDE.md

pause
