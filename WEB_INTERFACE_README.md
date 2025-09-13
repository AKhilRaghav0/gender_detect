# 🌐 Gender Detection Web Interface

A beautiful, responsive web interface for real-time gender detection using InsightFace and Flask.

## ✨ Features

- 🎥 **Real-time camera streaming** - Live video feed in your browser
- 🎯 **Face detection** - Automatic face detection using InsightFace SCRFD
- 👥 **Gender classification** - Real-time gender prediction with confidence scores
- 📱 **Mobile responsive** - Works perfectly on phones and tablets
- 🌐 **Network accessible** - Access from any device on your network
- 📸 **Image capture** - Save detection results as images
- 📊 **Live statistics** - Real-time FPS and detection counts
- 🎨 **Beautiful UI** - Modern, gradient-based interface

## 🚀 Quick Start

### 1. Start the Web App
```bash
# Make sure you're in the project directory
cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect

# Run the startup script
./start_web_app.sh
```

### 2. Access the Interface
- **Local**: http://localhost:5000
- **Network**: http://YOUR_IP:5000
- **Phone**: http://YOUR_IP:5000 (same network)

### 3. Use the Interface
1. Click "▶️ Start Detection" to begin
2. Allow camera access when prompted
3. View real-time gender detection results
4. Use "📸 Capture Image" to save results

## 🛠️ Manual Setup

If the startup script doesn't work:

```bash
# Create virtual environment
python3 -m venv gender_detection_env
source gender_detection_env/bin/activate

# Install requirements
pip install -r requirements.txt

# Start the app
python3 web_app.py
```

## 📱 Mobile Access

1. **Find your IP address:**
   ```bash
   hostname -I
   ```

2. **Access from phone:**
   - Connect to same WiFi network
   - Open browser: `http://YOUR_IP:5000`
   - Allow camera access when prompted

## 🎯 How It Works

1. **Camera Capture**: Browser accesses camera via getUserMedia API
2. **Frame Processing**: Frames sent to Flask backend
3. **Face Detection**: InsightFace SCRFD detects faces
4. **Gender Classification**: Simple heuristic-based gender prediction
5. **Results Display**: Real-time results shown in beautiful UI

## 🔧 Technical Details

- **Backend**: Flask with threading for camera processing
- **Frontend**: Pure HTML/CSS/JavaScript (no frameworks)
- **Detection**: InsightFace + OpenCV
- **Streaming**: Base64 encoded JPEG frames
- **Real-time**: ~30 FPS processing

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, mobile
- **Live Statistics**: Face count, FPS, gender counts
- **Confidence Bars**: Visual confidence indicators
- **Status Indicators**: Running/stopped status
- **Modern Styling**: Gradient backgrounds, smooth animations

## 🐛 Troubleshooting

### Camera Not Working
- Check browser permissions
- Try different browser (Chrome recommended)
- Ensure camera is not used by other apps

### Network Access Issues
- Check firewall settings
- Ensure devices are on same network
- Try different port if 5000 is blocked

### Performance Issues
- Close other camera applications
- Reduce video quality in browser settings
- Check system resources

## 📁 File Structure

```
├── web_app.py              # Flask web server
├── templates/
│   └── index.html          # Web interface
├── backend/
│   └── insightface_gender_detection.py  # Detection logic
├── start_web_app.sh        # Startup script
└── requirements.txt        # Dependencies
```

## 🎉 Benefits Over Desktop App

- ✅ **No camera driver issues** - Browser handles camera access
- ✅ **Cross-platform** - Works on any device with browser
- ✅ **Mobile friendly** - Use from your phone
- ✅ **Network accessible** - Multiple devices can view
- ✅ **Easy deployment** - Just run one command
- ✅ **Beautiful UI** - Modern, responsive design

Perfect for demos, presentations, and multi-device access! 🚀

