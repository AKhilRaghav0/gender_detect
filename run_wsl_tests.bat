@echo off
echo 🧪 Running Model Tests in WSL
echo =============================

echo Starting WSL and running tests...
wsl bash -c "
cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect
source gender_detection_env/bin/activate
pip install insightface
python BusSaheli/backend/test_all_models_wsl.py
"

echo.
echo Tests completed! Check the output above.
pause
