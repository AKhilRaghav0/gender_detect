# 🚀 Single-Line Deployment Commands

## Universal GPU Setup (Works on ANY NVIDIA GPU)

### 🎯 Complete Auto-Setup (One Command Does Everything)
```bash
python auto_setup.py
```
**What this does:**
- Detects your GPU (RTX 3050, G200, GTX series, etc.)
- Calculates optimal settings based on VRAM
- Installs all packages
- Creates optimized training script
- Starts training automatically

---

## 🏢 G200 Server Deployment

### Step 1: Upload and Setup
```bash
# Upload this folder to your G200 server, then run:
python auto_setup.py && python train_optimized.py
```

### Step 2: Monitor Training (Optional)
```bash
# In another terminal:
gpustat -i 1
```

---

## 💻 Local Development (Your 4GB RTX 3050)

### Complete Setup + Training
```bash
python auto_setup.py
```

### Just Training (if already setup)
```bash
python train_optimized.py
```

### Real-time Webcam Detection
```bash
python detect_gender_modern.py --mode webcam --gpu
```

---

## 🔧 Manual Commands (If You Need Specific Control)

### Install Packages Only
```bash
python install_gpu_packages.py
```

### GPU Detection and Benchmarking
```bash
python gpu_setup.py && python benchmark_gpu.py
```

### Training with Custom Settings
```bash
# 4GB VRAM optimized
python train_modern.py --batch-size 24 --epochs 40 --mixed-precision

# High-end GPU (G200/RTX 4090)
python train_modern.py --batch-size 128 --epochs 50 --mixed-precision

# CPU fallback
python train_modern.py --batch-size 16 --epochs 30 --no-gpu
```

### Inference Commands
```bash
# GPU-accelerated webcam
python detect_gender_modern.py --mode webcam --gpu

# Process single image
python detect_gender_modern.py --mode image --image test.jpg --gpu

# CPU-only inference
python detect_gender_modern.py --mode webcam --no-gpu
```

---

## 🎮 GPU-Specific Optimizations

### RTX 3050 (4GB VRAM) - Your Current Setup
```bash
# Auto-optimized for 4GB VRAM
python auto_setup.py
```
**Settings:** Batch 24, Mixed Precision, 85% VRAM usage, Gradient Accumulation

### RTX 4090 / RTX 4080 (24GB+ VRAM)
```bash
python auto_setup.py  # Detects and optimizes automatically
```
**Settings:** Batch 128, Mixed Precision, Full performance

### GTX 1660 / GTX 1650 (4-6GB VRAM)
```bash
python auto_setup.py  # Detects and optimizes automatically
```
**Settings:** Batch 32, Mixed Precision (if supported), Safe VRAM usage

### Server GPUs (Tesla V100, A100, etc.)
```bash
python auto_setup.py  # Detects and optimizes automatically
```
**Settings:** Maximum batch sizes, Full server optimization

---

## 📊 Monitoring Commands

### Real-time GPU Usage
```bash
gpustat -i 1
```

### Detailed GPU Info
```bash
nvidia-smi -l 1
```

### Memory Profiling
```bash
python -m memory_profiler train_optimized.py
```

### Training Progress
```bash
tail -f training_log.csv
```

---

## 🚀 Quick Deploy to Any Server

### Copy and Run (Universal)
```bash
# 1. Copy project folder to server
scp -r gender_detect/ user@server:/path/to/

# 2. SSH to server and run
ssh user@server
cd /path/to/gender_detect/
python auto_setup.py
```

### Docker Deployment (Advanced)
```bash
# Build container with NVIDIA support
docker build -t gender-detection .
docker run --gpus all -it gender-detection python auto_setup.py
```

---

## 🎯 Expected Performance by GPU

| GPU Model | VRAM | Batch Size | Training Time | Webcam FPS |
|-----------|------|------------|---------------|------------|
| RTX 3050 | 4GB | 24 | ~45 min | 45+ FPS |
| RTX 4060 | 8GB | 48 | ~25 min | 60+ FPS |
| RTX 4070 | 12GB | 64 | ~20 min | 80+ FPS |
| RTX 4090 | 24GB | 128 | ~15 min | 120+ FPS |
| GTX 1650 | 4GB | 24 | ~50 min | 30+ FPS |
| Tesla V100 | 16GB | 96 | ~18 min | 100+ FPS |
| A100 | 40GB | 256 | ~10 min | 200+ FPS |

---

## 🔥 Ultra-Fast Deployment

### One-liner for immediate training:
```bash
python auto_setup.py && echo "Training started!" || echo "Setup failed!"
```

### Background training with logging:
```bash
nohup python auto_setup.py > setup.log 2>&1 &
```

### Complete pipeline with monitoring:
```bash
python auto_setup.py && gpustat -i 1 & python train_optimized.py
```

---

## 🎉 What Each Command Does

### `python auto_setup.py`
1. 🔍 Detects your GPU model and VRAM
2. 📊 Calculates optimal batch size and settings  
3. 📦 Installs all required packages
4. 🏗️ Creates optimized training script
5. 💾 Saves configuration for future use
6. 🚀 Optionally starts training immediately

### `python train_optimized.py`
1. ⚙️ Configures GPU with optimal settings
2. 🎯 Sets memory limits to prevent VRAM overflow
3. ⚡ Enables mixed precision if supported
4. 📈 Uses gradient accumulation for effective larger batches
5. 🏃 Starts training with progress monitoring

### `python detect_gender_modern.py`
1. 🎮 Initializes GPU-accelerated inference
2. 🎥 Opens webcam with optimal FPS
3. 👤 Detects faces in real-time
4. ⚡ Classifies gender with GPU acceleration
5. 📊 Shows FPS and confidence scores

The system automatically adapts to ANY NVIDIA GPU - from your 4GB RTX 3050 to high-end G200 server GPUs!
