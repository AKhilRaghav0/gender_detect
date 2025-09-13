#!/bin/bash
# WSL Ubuntu Setup Script for InsightFace Gender Detection

echo "🚀 Setting up Ubuntu WSL2 for InsightFace Gender Detection..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and pip
echo "🐍 Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies for OpenCV
echo "📷 Installing OpenCV dependencies..."
sudo apt install -y libopencv-dev python3-opencv

# Install other system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Create virtual environment
echo "🌐 Creating virtual environment..."
python3 -m venv gender_detection_env
source gender_detection_env/bin/activate

# Install Python packages
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install insightface onnxruntime opencv-python numpy Pillow

# Install additional packages for better performance
echo "⚡ Installing performance packages..."
pip install onnx

echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "source gender_detection_env/bin/activate"
echo ""
echo "To run the detection:"
echo "python backend/insightface_gender_detection.py"
echo ""
echo "Note: Camera access in WSL requires X11 forwarding or WSLg"

