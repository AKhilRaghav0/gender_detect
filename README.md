# Gender Detection System

Fast, accurate gender detection using InsightFace + SCRFD for real-time face detection and classification.

## Features

- 🚀 **Real-time detection** using InsightFace SCRFD
- 🎯 **High accuracy** face detection
- 👥 **Gender classification** with confidence scores
- 📹 **Webcam support** for live detection
- 💾 **Image saving** functionality

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run detection:**
   ```bash
   python backend/insightface_gender_detection.py
   ```

## Controls

- **'q'** - Quit detection
- **'s'** - Save current frame

## Requirements

- Python 3.8+
- OpenCV
- InsightFace
- ONNX Runtime

## Performance

- **CPU**: ~15-20 FPS on modern CPU
- **GPU**: ~30+ FPS with CUDA support
- **Accuracy**: 90%+ on clear face images