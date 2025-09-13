@echo off
echo Starting Ubuntu WSL and setting up gender detection...
echo.

REM Start WSL and run the setup script
wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect && chmod +x setup_wsl.sh && ./setup_wsl.sh"

echo.
echo Setup complete! To run detection in WSL:
echo wsl -d Ubuntu-22.04 bash -c "cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect && source gender_detection_env/bin/activate && python backend/insightface_gender_detection.py"
pause
