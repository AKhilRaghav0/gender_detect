# 🚀 Gender Detection Project - VPS Deployment Guide

## 📋 Prerequisites
- Ubuntu/Debian VPS with at least 4GB RAM
- SSH access to your VPS
- Your dataset (`gender_dataset_face/` folder)

## 🚀 Quick Start

### 1. Upload Project to VPS
```bash
# From your local machine, upload the project
scp -r ./gender_dataset_face/ user@your-vps-ip:/home/user/
scp -r ./* user@your-vps-ip:/home/user/gender-detection/
```

### 2. SSH into VPS
```bash
ssh user@your-vps-ip
cd gender-detection
```

### 3. Run Installation Script
```bash
bash install_vps.sh
```

### 4. Test Setup
```bash
python3 test_setup.py
```

### 5. Start Training
```bash
python3 train_vps.py
```

## 📊 Training Optimization

The VPS training script is optimized for:
- **Reduced epochs**: 50 instead of 100 (faster training)
- **Smaller batch size**: 32 instead of 64 (less memory usage)
- **Progress tracking**: Shows processing progress every 100 images
- **Performance monitoring**: Tracks training time and results
- **Non-interactive plotting**: Uses 'Agg' backend for VPS environment

## 🔧 Manual Installation (Alternative)

If you prefer manual installation:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-pip python3-venv

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv libgl1-mesa-glx

# Create virtual environment
python3 -m venv gender_detection_env
source gender_detection_env/bin/activate

# Install Python packages
pip install tensorflow==2.15.0 opencv-python==4.7.0.72 cvlib==0.2.7 scikit-learn matplotlib numpy
```

## 📁 Project Structure on VPS

```
gender-detection/
├── gender_dataset_face/          # Your dataset
│   ├── man/                     # Male face images
│   └── woman/                   # Female face images
├── install_vps.sh               # Installation script
├── train_vps.py                 # VPS-optimized training script
├── detect_gender_vps.py         # VPS webcam detection
├── test_setup.py                # Setup testing script
├── requirements.txt              # Python dependencies
└── README.md                    # Project documentation
```

## 🎯 Training Commands

### Start Training
```bash
python3 train_vps.py
```

### Monitor Training
- Training progress is displayed in real-time
- Progress bars show current epoch and accuracy
- Final results are saved to `training_results_vps.png`

### Training Output
- **Model**: `gender_detection_vps.model`
- **Plots**: `training_results_vps.png`
- **Console**: Real-time training metrics

## 📹 Running Detection

### Webcam Detection
```bash
python3 detect_gender_vps.py
```

### Features
- Real-time face detection
- Gender classification with confidence scores
- Performance monitoring (FPS, frame count)
- Press 'Q' to quit

## 🔍 Troubleshooting

### Common Issues

1. **OpenCV Installation Failed**
   ```bash
   sudo apt install -y libopencv-dev python3-opencv
   ```

2. **Memory Issues During Training**
   - Reduce batch size in `train_vps.py`
   - Close other applications
   - Use swap memory if needed

3. **Webcam Not Working**
   - Ensure VPS has webcam access
   - Check USB device permissions
   - Try different device indices (0, 1, 2)

4. **Model Loading Failed**
   - Ensure training completed successfully
   - Check file permissions
   - Verify model file exists

### Performance Tips

- **GPU Training**: If available, TensorFlow will automatically use GPU
- **Memory Management**: Monitor memory usage with `htop`
- **Background Training**: Use `nohup` for long training sessions
- **Logging**: Redirect output to log files for monitoring

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Watch training progress
tail -f training.log

# Monitor system resources
htop
```

### Training Logs
```bash
# Start training with logging
nohup python3 train_vps.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## 🎉 Success Indicators

✅ **Installation Complete**: All packages imported successfully  
✅ **Training Started**: Model building and training begins  
✅ **Training Complete**: Final accuracy and loss displayed  
✅ **Model Saved**: `gender_detection_vps.model` created  
✅ **Detection Working**: Webcam opens and processes frames  

## 🚀 Next Steps

After successful deployment:
1. **Train your model** with custom dataset
2. **Test detection** on webcam
3. **Optimize parameters** for better accuracy
4. **Deploy to production** if needed
5. **Monitor performance** and retrain as needed

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check system resources (CPU, RAM, disk)
4. Review error messages in console output

---

**Happy Training! 🎯🚀**
