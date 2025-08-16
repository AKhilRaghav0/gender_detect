# 🚀 Lambda Cloud Deployment Guide

## **Quick Start (One Command)**

```bash
# Clone your repo on Lambda Cloud
git clone <your-repo-url>
cd gender_detect

# Run the deployment script
bash deploy.sh
```

## **Manual Setup (Step by Step)**

### **1. Install Dependencies**
```bash
pip install -r requirements_cloud.txt
```

### **2. Verify GPU Detection**
```bash
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU devices: {len(gpus)}')"
```

### **3. Start Training**
```bash
python train_modern.py
```

## **What You'll Get**

### **✅ Automatic GPU Detection**
- RTX 4090 (24GB VRAM) - **6x faster** than your RTX 3050
- RTX 3090 (24GB VRAM) - **5x faster** than your RTX 3050
- A100 (40GB+ VRAM) - **10x faster** than your RTX 3050

### **✅ No Setup Issues**
- CUDA pre-installed
- Drivers optimized
- Python 3.11 ready
- All dependencies compatible

### **✅ Training Performance**
- **Local RTX 3050**: 6-8 hours
- **Lambda Cloud**: 1-2 hours
- **Cost**: ~$0.60-1.20/hour

## **Monitoring Training**

### **TensorBoard (Real-time)**
```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

### **Check Progress**
```bash
# View training logs
tail -f training_logs/training.log

# Check GPU usage
nvidia-smi
```

## **Download Trained Model**

After training completes:
```bash
# Your model will be saved as:
# - gender_detection_modern.keras (main model)
# - model_config.json (configuration)
# - training_history.png (training curves)
```

## **Files to Run**

### **🎯 Main Training Script**
```bash
python train_modern.py
```

### **🔍 GPU Test Script**
```bash
python test_gpu.py
```

### **📊 Benchmark Script**
```bash
python benchmark_gpu.py
```

### **🚀 Auto-Deploy Script**
```bash
python deploy_cloud.py
```

## **Expected Output**

```
🚀 Modern Gender Detection Training
📁 Data directory: gender_dataset_face
🖼️  Image size: 224x224
📦 Batch size: 32
🔄 Epochs: 50
✅ GPU configured: 1 GPU(s) available
  GPU 0: NVIDIA GeForce RTX 4090
🚀 RTX 4090 detected - Enabling mixed precision
```

## **Troubleshooting**

### **If GPU Not Detected**
```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Reinstall TensorFlow
pip install --force-reinstall tensorflow
```

### **If Training Fails**
```bash
# Check logs
cat training_logs/error.log

# Verify dataset structure
ls -la gender_dataset_face/
```

## **Cost Optimization**

### **Best GPU for Your Use Case**
- **RTX 4090**: Fastest, ~$1.20/hour
- **RTX 3090**: Balanced, ~$0.80/hour
- **A100**: Most powerful, ~$2.00/hour

### **Estimated Total Cost**
- **Training time**: 1-2 hours
- **Total cost**: $0.80-$2.40
- **vs Local setup**: Priceless (saves 4-6 hours of debugging)

## **Next Steps After Training**

1. **Download model** to your local machine
2. **Test inference** with `python detect_gender_modern.py`
3. **Deploy to production** (Raspberry Pi 5, web server, etc.)
4. **Monitor performance** and retrain if needed

---

**🎉 You're all set! The cloud will handle all the GPU complexity for you.**
