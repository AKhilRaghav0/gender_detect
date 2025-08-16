#!/usr/bin/env python3
"""
Training Status Checker
Check if training is running and monitor progress
"""

import subprocess
import time
from pathlib import Path
import pandas as pd

def check_training_process():
    """Check if training process is running"""
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        python_processes = [line for line in result.stdout.split('\n') if 'python' in line.lower()]
        
        if python_processes:
            print("Python processes running:")
            for proc in python_processes:
                if proc.strip():
                    print(f"  {proc.strip()}")
            return True
        else:
            print("No Python processes found")
            return False
    except:
        print("Could not check processes")
        return False

def check_training_files():
    """Check training progress files"""
    print("\nTraining Files Status:")
    
    files_to_check = [
        'training_log_simple.csv',
        'best_gender_model_simple.keras', 
        'gender_detection_simple.keras',
        'training_history_simple.png'
    ]
    
    for file in files_to_check:
        path = Path(file)
        if path.exists():
            size = path.stat().st_size
            mtime = time.ctime(path.stat().st_mtime)
            print(f"  ✅ {file} ({size} bytes, modified: {mtime})")
        else:
            print(f"  ⏳ {file} (not found)")

def show_training_progress():
    """Show training progress from CSV log"""
    log_file = Path('training_log_simple.csv')
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            if len(df) > 0:
                print(f"\nTraining Progress ({len(df)} epochs completed):")
                latest = df.iloc[-1]
                print(f"  Latest Epoch: {latest['epoch'] + 1}")
                print(f"  Training Accuracy: {latest['accuracy']:.4f}")
                print(f"  Validation Accuracy: {latest['val_accuracy']:.4f}")
                print(f"  Training Loss: {latest['loss']:.4f}")
                print(f"  Validation Loss: {latest['val_loss']:.4f}")
                
                if len(df) > 1:
                    best_val_acc = df['val_accuracy'].max()
                    best_epoch = df['val_accuracy'].idxmax() + 1
                    print(f"  Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
            else:
                print("\nTraining log is empty - training may have just started")
        except Exception as e:
            print(f"\nError reading training log: {e}")
    else:
        print("\nNo training log found yet")

def main():
    """Main status check function"""
    print("=" * 50)
    print("Training Status Check")
    print("=" * 50)
    
    # Check if training is running
    is_running = check_training_process()
    
    # Check training files
    check_training_files()
    
    # Show progress
    show_training_progress()
    
    print("\n" + "=" * 50)
    if is_running:
        print("✅ Training appears to be running!")
        print("💡 Tips:")
        print("  - Run this script again to check progress")
        print("  - Training typically takes 30-60 minutes on RTX 3050")
        print("  - Check GPU usage: nvidia-smi")
    else:
        print("❌ No training process detected")
        print("💡 To start training:")
        print("  python train_simple.py")

if __name__ == "__main__":
    main()
