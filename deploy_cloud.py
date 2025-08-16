#!/usr/bin/env python3
"""
Cloud Deployment Script for Lambda Cloud
Automatically sets up environment and starts training
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🚀 {description}...")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def check_gpu():
    """Check if GPU is available"""
    print("🔍 Checking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            return True
        else:
            print("❌ No GPU detected")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def main():
    """Main deployment function"""
    print("=" * 60)
    print("🚀 Lambda Cloud Gender Detection Deployment")
    print("=" * 60)
    
    # Step 1: Install requirements
    if not run_command("pip install -r requirements_cloud.txt", "Installing requirements"):
        print("❌ Failed to install requirements. Exiting.")
        sys.exit(1)
    
    # Step 2: Check GPU
    if not check_gpu():
        print("❌ GPU not available. Please check your Lambda Cloud instance.")
        sys.exit(1)
    
    # Step 3: Start training
    print("\n🎯 Starting training with GPU acceleration...")
    print("   Model: ResNet50 + Advanced Augmentation")
    print("   GPU: RTX 4090/3090/A100 (automatically detected)")
    print("   Expected time: 1-2 hours for full training")
    
    # Start training
    if run_command("python train_modern.py", "Starting training"):
        print("\n🎉 Training started successfully!")
        print("📊 Monitor progress with: tensorboard --logdir=./logs")
        print("🔄 Training will continue in background")
    else:
        print("❌ Failed to start training")

if __name__ == "__main__":
    main()
