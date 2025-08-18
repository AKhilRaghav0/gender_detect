# 🚀 Gender Detection Inference Guide

## 🎯 Overview
You've already trained your models on the GH200! Now let's run them on your laptop for real-time gender detection. **No GPU training needed** - just lightweight inference! 🎉

## 📋 What You Have
- ✅ **Trained Models**: `gender_detection_modern.keras` (113MB)
- ✅ **Inference Script**: `run_gender_detection.py`
- ✅ **Lightweight Requirements**: `requirements_inference.txt`

## 🚀 Quick Start

### 1. **Install Lightweight Requirements**
```bash
pip install -r requirements_inference.txt
```

### 2. **Run Real-Time Webcam Detection**
```bash
python run_gender_detection.py
```
- **Press 'q'** to quit
- **Press 's'** to save current frame
- **Green boxes** = Men
- **Pink boxes** = Women

### 3. **Test on Single Image**
```bash
python run_gender_detection.py --mode image --input your_image.jpg
```

### 4. **Batch Process Folder**
```bash
python run_gender_detection.py --mode batch --input your_images_folder/
```

## 🎮 **Usage Examples**

### **Webcam Mode (Default)**
```bash
# Start webcam detection
python run_gender_detection.py

# Use specific model
python run_gender_detection.py --model best_gender_model.keras
```

### **Single Image Mode**
```bash
# Test on one image
python run_gender_detection.py --mode image --input test_photo.jpg

# Use different model
python run_gender_detection.py --mode image --input test_photo.jpg --model gender_detection.model
```

### **Batch Mode**
```bash
# Process all images in folder
python run_gender_detection.py --mode batch --input photos/

# Process with specific model
python run_gender_detection.py --mode batch --input photos/ --model best_gender_model.keras
```

## 📊 **What You'll See**

### **Webcam Mode:**
- Real-time face detection
- Gender prediction with confidence
- Color-coded boxes (Green=Man, Pink=Woman)
- FPS counter
- Save functionality

### **Image Mode:**
- Face detection in single image
- Gender labels on each face
- Annotated image saved
- Confidence scores displayed

### **Batch Mode:**
- Process multiple images
- Summary statistics
- Count of men vs women
- Individual results for each image

## 🔧 **Troubleshooting**

### **Model Loading Issues:**
```bash
# Check available models
ls -la *.keras *.model

# Try different model
python run_gender_detection.py --model best_gender_model.keras
```

### **Webcam Issues:**
```bash
# Test webcam access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam available:', cap.isOpened())"
```

### **Performance Issues:**
```bash
# Install CPU-only TensorFlow (lighter)
pip uninstall tensorflow
pip install tensorflow-cpu
```

## 💡 **Pro Tips**

1. **Start with webcam mode** to test everything works
2. **Use 's' key** to save interesting detections
3. **Check confidence scores** - higher is better
4. **Good lighting** improves face detection
5. **Face the camera** for best results

## 🎯 **Expected Performance**

- **Accuracy**: 83% (as trained on GH200)
- **Speed**: 15-30 FPS on modern laptop
- **Memory**: ~200-300MB RAM usage
- **CPU**: Works on any modern processor

## 🚀 **Advanced Usage**

### **Custom Model Path:**
```bash
python run_gender_detection.py --model /path/to/your/model.keras
```

### **Different Input Sources:**
```bash
# Video file
python run_gender_detection.py --mode image --input video.mp4

# Network camera
python run_gender_detection.py --mode image --input rtsp://camera_url
```

## 🎉 **You're All Set!**

Your **83% accurate** gender detection model is ready to run on your laptop! 

- **No more GPU setup issues**
- **No more training time**
- **Just instant gender detection!**

**🎯 Ready to detect some genders? Run the webcam mode and see it in action! 🚀**

