#!/bin/bash
# Start Gender Detection Web App

echo "🚀 Starting Gender Detection Web App..."

# Check if we're in WSL
if grep -q Microsoft /proc/version; then
    echo "🐧 Running in WSL - Linux environment detected"
else
    echo "💻 Running on native Linux"
fi

# Check if virtual environment exists
if [ ! -d "gender_detection_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv gender_detection_env
fi

# Activate virtual environment
echo "🌐 Activating virtual environment..."
source gender_detection_env/bin/activate

# Install/update requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Get local IP address
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "🌐 Web Interface URLs:"
echo "   Local:  http://localhost:5000"
echo "   Network: http://$LOCAL_IP:5000"
echo ""
echo "📱 Access from phone: http://$LOCAL_IP:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the web app
python3 web_app.py

