import os
import sys
import subprocess

def main():
    print("Starting 4GB GPU optimized training...")
    
    python_exe = "gender_detection_env/Scripts/python.exe"
    
    # Simple training command with 4GB optimizations
    cmd = [
        python_exe,
        "train_modern.py"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Set environment variables for 4GB GPU optimization
    env = os.environ.copy()
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    env['TF_GPU_MEMORY_LIMIT'] = '3400'  # 3.4GB limit
    
    try:
        subprocess.run(cmd, env=env)
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
