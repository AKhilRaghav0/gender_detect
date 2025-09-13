# WSL2 Setup Status

## Current Status: 🔄 Installing Ubuntu

Ubuntu WSL2 is currently downloading and installing. This may take a few minutes.

## Next Steps:

1. **Wait for Ubuntu installation to complete**
2. **Run the setup script:**
   ```bash
   # Double-click this file or run in PowerShell:
   run_in_wsl.bat
   ```

3. **Or manually in WSL:**
   ```bash
   wsl -d Ubuntu
   cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect
   chmod +x setup_wsl.sh
   ./setup_wsl.sh
   ```

## What the setup does:

- ✅ Installs Python 3 and pip
- ✅ Installs OpenCV system dependencies  
- ✅ Creates virtual environment
- ✅ Installs InsightFace and all requirements
- ✅ Sets up camera access for WSL

## After setup:

```bash
# Activate environment
source gender_detection_env/bin/activate

# Run detection
python backend/insightface_gender_detection.py
```

## Camera Access in WSL:

- **WSLg** (Windows 11): Should work automatically
- **X11 forwarding** (Windows 10): May need additional setup
- **Alternative**: Use Windows version for camera, WSL for processing

## Files Created:

- `setup_wsl.sh` - Ubuntu setup script
- `run_in_wsl.bat` - Windows batch file to run setup
- `backend/insightface_gender_detection.py` - Main detection script
- `requirements.txt` - Python dependencies

