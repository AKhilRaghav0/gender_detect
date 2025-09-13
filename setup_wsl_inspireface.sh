#!/bin/bash

# InspireFace WSL Setup Script
# Run this in WSL Ubuntu environment

echo "🚀 Setting up InspireFace in WSL"
echo "================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🔧 Installing development tools..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    build-essential \
    git \
    wget \
    unzip \
    libopencv-dev \
    libeigen3-dev \
    pkg-config

# Install Python packages
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install numpy opencv-python torch torchvision

# Clone InspireFace repository
echo "📥 Cloning InspireFace repository..."
if [ ! -d "InspireFace" ]; then
    git clone https://github.com/HyperInspire/InspireFace.git
else
    echo "InspireFace directory already exists, updating..."
    cd InspireFace
    git pull
    cd ..
fi

# Setup InspireFace
echo "🔧 Setting up InspireFace..."
cd InspireFace

# Install Python dependencies
pip3 install -r requirements.txt

# Download CPU model (lightweight)
echo "⬇️ Downloading CPU model..."
MODEL_URL="https://github.com/HyperInspire/InspireFace/releases/download/v1.2.3/inspireface-linux-x86-manylinux2014-1.2.3.zip"
wget -O model.zip $MODEL_URL

# Extract model
echo "📦 Extracting model..."
unzip model.zip
rm model.zip

# Build C++ extensions (if needed)
echo "🔨 Building extensions..."
if [ -f "setup.py" ]; then
    pip3 install -e .
elif [ -f "pyproject.toml" ]; then
    pip3 install .
fi

# Create test script
cat > test_inspireface.py << 'EOF'
#!/usr/bin/env python3
"""
Test InspireFace installation
"""

try:
    import inspireface as ifa
    print("✅ InspireFace imported successfully!")
    print("🎯 Version:", ifa.__version__ if hasattr(ifa, '__version__') else "Unknown")

    # Test basic functionality
    analyzer = ifa.InspireFaceAnalyzer()
    print("✅ Analyzer created!")

except ImportError as e:
    print("❌ InspireFace import failed:", e)
    print("💡 Make sure the model files are in the correct location")

except Exception as e:
    print("❌ Test failed:", e)
EOF

# Test installation
echo "🧪 Testing InspireFace..."
python3 test_inspireface.py

echo ""
echo "🎉 InspireFace setup complete!"
echo "=============================="
echo "📍 Repository: $(pwd)"
echo "📦 Model: $(pwd)/inspireface-linux-x86-manylinux2014-1.2.3"
echo ""
echo "🚀 To test: python3 test_inspireface.py"
echo "📚 For full integration guide: ../INSPIREFACE_INTEGRATION_GUIDE.md"
echo ""
echo "💡 Next steps:"
echo "1. Copy your gender detection project to WSL"
echo "2. Follow the integration guide"
echo "3. Enjoy professional-grade face analysis!"
