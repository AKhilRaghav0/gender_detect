# 🚀 Lambda Cloud GH200 Training Guide

## 🎯 Overview
This guide will help you run all the advanced training modes on your Lambda Cloud GH200 instance. Your GH200 has **480GB VRAM** - let's make it sweat! 💪

## 📋 Prerequisites
- ✅ Lambda Cloud instance with GH200 GPU
- ✅ All training scripts uploaded
- ✅ Dataset (`gender_dataset_face/`) available
- ✅ Python environment with TensorFlow installed

## 🚀 Quick Start

### 1. **Pull Latest Changes**
```bash
git pull
```

### 2. **Run Master Training Script**
```bash
python run_all_training_modes.py
```

### 3. **Choose Your Training Mode**
The script will show you 5 options:

## 📚 Training Modes Available

### 🎯 **Mode 1: Standard Training** (15-30 min)
- **Script**: `train_modern_fixed.py`
- **Best for**: Quick results, baseline performance
- **What it does**: Basic ResNet50 training with your current dataset
- **Expected accuracy**: 85-95%

### 🔄 **Mode 2: K-Fold Cross-Validation** (2-4 hours)
- **Script**: `train_kfold_robust.py`
- **Best for**: Production deployment, maximum reliability
- **What it does**: 5-fold CV with 3 different architectures (ResNet50, EfficientNet, DenseNet)
- **Expected accuracy**: 90-98% with confidence intervals
- **Output**: 5 models + ensemble results

### 🎨 **Mode 3: Advanced Augmentation** (1-2 hours)
- **Script**: `train_advanced_augmentation.py`
- **Best for**: Limited data, maximum generalization
- **What it does**: Intensive data augmentation (3x dataset size) + EfficientNet
- **Expected accuracy**: 88-96%
- **Features**: 3 levels of augmentation (basic, advanced, extreme)

### 🏗️ **Mode 4: Multi-Model Ensemble** (4-6 hours)
- **Script**: `train_ensemble_models.py`
- **Best for**: Best accuracy, research, competition
- **What it does**: Train 6 different architectures and combine them
- **Models**: ResNet50, EfficientNet, DenseNet, InceptionV3, Xception, MobileNetV2
- **Expected accuracy**: 92-99%
- **Output**: 6 individual models + ensemble model

### 🔥 **Mode 5: Run All Modes** (8-12 hours)
- **What it does**: Run all modes sequentially
- **Best for**: Complete evaluation, maximum learning
- **Output**: All models from all modes + comprehensive results

## 💡 **Recommended Strategy**

### **For Quick Results:**
```bash
python train_modern_fixed.py
```

### **For Production Use:**
```bash
python train_kfold_robust.py
```

### **For Maximum Performance:**
```bash
python train_ensemble_models.py
```

### **For Research/Competition:**
```bash
python run_all_training_modes.py
# Choose option 5
```

## 🎮 **Manual Script Execution**

If you prefer to run scripts directly:

### **Standard Training:**
```bash
python train_modern_fixed.py
```

### **K-Fold Training:**
```bash
python train_kfold_robust.py
```

### **Advanced Augmentation:**
```bash
python train_advanced_augmentation.py
```

### **Multi-Model Ensemble:**
```bash
python train_ensemble_models.py
```

## 📊 **Expected Outputs**

### **Files Generated:**
- `gender_detection_*.keras` - Trained models
- `best_*_model.keras` - Best checkpoints
- `*_training_results.png` - Training curves
- `*_results.json` - Performance metrics
- `logs/` - TensorBoard logs

### **Performance Metrics:**
- **Accuracy**: 85-99% depending on mode
- **Precision**: 85-99%
- **Recall**: 85-99%
- **F1-Score**: 85-99%

## 🔍 **Monitoring Training**

### **Real-time Monitoring:**
```bash
# In another terminal
tensorboard --logdir=./logs
```

### **GPU Usage:**
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### **Training Progress:**
- Watch the terminal output
- Check generated plots
- Monitor GPU utilization

## ⚡ **GH200 Optimizations**

All scripts automatically:
- ✅ Enable mixed precision (float16)
- ✅ Enable XLA compilation
- ✅ Use full 480GB VRAM
- ✅ Optimize batch sizes
- ✅ Enable memory growth

## 🚨 **Troubleshooting**

### **Out of Memory:**
- Scripts automatically handle this
- GH200 has 480GB - should be fine

### **Training Stuck:**
- Check GPU utilization: `nvidia-smi`
- Monitor logs for errors
- Restart if necessary

### **Model Not Saving:**
- Check disk space: `df -h`
- Ensure write permissions

## 💰 **Cost Optimization**

### **Estimated Costs (GH200):**
- **Mode 1**: $0.10-0.20
- **Mode 2**: $0.50-1.00
- **Mode 3**: $0.30-0.60
- **Mode 4**: $1.00-2.00
- **Mode 5**: $2.00-4.00

### **Cost Saving Tips:**
- Use early stopping (built-in)
- Monitor training progress
- Stop if accuracy plateaus
- Use TensorBoard to track progress

## 🎉 **Success Indicators**

### **Good Training:**
- Accuracy increasing steadily
- Validation accuracy following training
- Loss decreasing
- GPU utilization >80%

### **Training Complete:**
- Early stopping triggered
- Best model saved
- Performance metrics displayed
- Plots generated

## 🚀 **Next Steps After Training**

### **Test Your Models:**
```bash
# Create a simple test script
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('gender_detection_*.keras')
print('Model loaded successfully!')
"
```

### **Download Models:**
```bash
# Download to your local machine
scp -r ubuntu@YOUR_INSTANCE_IP:~/gender_detect/*.keras ./
```

### **Deploy:**
- Use models in your applications
- Integrate with web services
- Deploy to edge devices

## 🔥 **Pro Tips**

1. **Start with Mode 1** to verify everything works
2. **Use Mode 2** for production deployment
3. **Try Mode 4** if you want maximum accuracy
4. **Monitor GPU utilization** to ensure full usage
5. **Save models frequently** (automatic with checkpoints)
6. **Use TensorBoard** for detailed monitoring

## 🆘 **Need Help?**

### **Check Scripts Exist:**
```bash
ls -la *.py
```

### **Check Python Environment:**
```bash
python --version
pip list | grep tensorflow
```

### **Check GPU:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

**🎯 Ready to make your GH200 sweat? Choose your training mode and let's go! 🚀**
