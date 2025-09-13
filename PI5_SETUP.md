# Raspberry Pi 5 Gender Detection Setup

## 🍓 **Optimized for 24/7 Bus Operation**

### **Hardware Requirements:**
- Raspberry Pi 5 (4GB+ RAM recommended)
- Camera module or USB webcam
- MicroSD card (32GB+ Class 10)
- Power supply (5V 3A+)

### **Performance Optimizations Applied:**

#### **1. Model Optimizations:**
- **Buffalo_S model** - Smallest, fastest InsightFace model
- **256x256 detection** - Minimal processing size
- **CPU-only execution** - No GPU dependencies
- **Max 2 faces** - Reduced processing load

#### **2. Image Processing:**
- **480px max width** - Reduced image size
- **20% JPEG quality** - Minimal bandwidth
- **2 FPS processing** - Stable performance
- **Memory cleanup** - Every 100 frames

#### **3. System Optimizations:**
- **Performance CPU governor** - Maximum processing power
- **128MB GPU memory** - Optimized memory split
- **High priority process** - Better resource allocation
- **Auto-restart** - Crash recovery

### **Installation:**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip3 install -r requirements_pi5.txt

# Set up camera
sudo raspi-config
# Enable Camera Interface

# Make startup script executable
chmod +x start_pi5.sh

# Start the application
./start_pi5.sh
```

### **Access:**
- **Local**: http://localhost:5000
- **Network**: http://[PI_IP]:5000
- **Mobile**: Use network IP on phone

### **24/7 Operation:**
- **Auto-restart** on crashes
- **Memory management** prevents leaks
- **Low power consumption** optimized
- **Stable 2 FPS** performance

### **Expected Performance:**
- **2 FPS** - Stable for bus monitoring
- **80%+ accuracy** - Reliable gender detection
- **<1GB RAM** - Efficient memory usage
- **<50% CPU** - Leaves resources for other tasks

### **Troubleshooting:**
- Check camera permissions
- Verify network connectivity
- Monitor system resources
- Check logs for errors

