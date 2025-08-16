#!/usr/bin/env python3
"""
GPU Setup and Optimization for Gender Detection
Supports NVIDIA RTX 3050, GTX 1200 series, and other CUDA-compatible GPUs
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_nvidia_gpu():
    """Check for NVIDIA GPU and CUDA support"""
    print("🔍 Checking NVIDIA GPU availability...")
    
    try:
        # Try nvidia-smi command
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            
            # Parse GPU information
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                    gpu_info = line.strip()
                    print(f"🎮 GPU: {gpu_info}")
                    
                    # Detect specific GPU models
                    if 'RTX 3050' in gpu_info:
                        print("🚀 RTX 3050 detected - Excellent for AI training!")
                        return 'RTX_3050'
                    elif 'GTX 12' in gpu_info:
                        print("🚀 GTX 1200 series detected - Good for AI training!")
                        return 'GTX_1200'
                    elif 'RTX' in gpu_info:
                        print("🚀 RTX series detected - Excellent for AI training!")
                        return 'RTX'
                    elif 'GTX' in gpu_info:
                        print("🚀 GTX series detected - Good for AI training!")
                        return 'GTX'
                    else:
                        return 'NVIDIA_OTHER'
            
            return 'NVIDIA_UNKNOWN'
        else:
            print("❌ nvidia-smi command failed")
            return None
            
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return None
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi timeout")
        return None
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return None

def check_cuda_version():
    """Check CUDA version"""
    print("\n🔍 Checking CUDA version...")
    
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"✅ CUDA: {line.strip()}")
                    
                    # Extract CUDA version
                    if 'V11.' in line:
                        return '11.x'
                    elif 'V12.' in line:
                        return '12.x'
                    else:
                        return 'unknown'
        else:
            print("❌ nvcc command failed")
            return None
            
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA toolkit may not be installed")
        return None
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return None

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n🔍 Checking TensorFlow GPU support...")
    
    try:
        import tensorflow as tf
        
        print(f"📦 TensorFlow version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow can see {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
                # Get GPU details
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        print(f"    Details: {gpu_details}")
                except Exception as e:
                    print(f"    Could not get details: {e}")
            
            return True
        else:
            print("❌ TensorFlow cannot see any GPUs")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking TensorFlow GPU: {e}")
        return False

def configure_gpu_memory():
    """Configure GPU memory growth to prevent VRAM issues"""
    print("\n⚙️  Configuring GPU memory...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU memory growth enabled")
                
                # Optional: Set memory limit (uncomment if needed)
                # tf.config.experimental.set_memory_limit(gpus[0], 4096)  # 4GB limit
                
                return True
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
                return False
        else:
            print("ℹ️  No GPUs to configure")
            return False
            
    except Exception as e:
        print(f"❌ Error configuring GPU: {e}")
        return False

def install_gpu_packages(cuda_version=None):
    """Install GPU-specific packages"""
    print(f"\n📦 Installing GPU packages for CUDA {cuda_version}...")
    
    packages = [
        'psutil',
        'memory-profiler',
        'pynvml',
        'gpustat'
    ]
    
    # Add CUDA-specific packages if available
    if cuda_version == '11.x':
        packages.append('cupy-cuda11x')
    elif cuda_version == '12.x':
        packages.append('cupy-cuda12x')
    
    python_exe = sys.executable
    
    for package in packages:
        try:
            print(f"  Installing {package}...")
            result = subprocess.run([python_exe, '-m', 'pip', 'install', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ {package} installed")
            else:
                print(f"  ⚠️  {package} installation failed: {result.stderr}")
        except Exception as e:
            print(f"  ❌ Error installing {package}: {e}")

def create_gpu_config():
    """Create GPU configuration file"""
    config = {
        "gpu_available": False,
        "gpu_type": None,
        "cuda_version": None,
        "tensorflow_gpu": False,
        "recommended_batch_size": 32,
        "recommended_image_size": 224,
        "mixed_precision": False
    }
    
    # Check GPU
    gpu_type = check_nvidia_gpu()
    if gpu_type:
        config["gpu_available"] = True
        config["gpu_type"] = gpu_type
        
        # Set recommendations based on GPU type
        if gpu_type == 'RTX_3050':
            config["recommended_batch_size"] = 64
            config["recommended_image_size"] = 224
            config["mixed_precision"] = True
        elif gpu_type == 'GTX_1200':
            config["recommended_batch_size"] = 32
            config["recommended_image_size"] = 224
            config["mixed_precision"] = False
        elif 'RTX' in gpu_type:
            config["recommended_batch_size"] = 128
            config["recommended_image_size"] = 256
            config["mixed_precision"] = True
        elif 'GTX' in gpu_type:
            config["recommended_batch_size"] = 32
            config["recommended_image_size"] = 224
            config["mixed_precision"] = False
    
    # Check CUDA
    cuda_version = check_cuda_version()
    if cuda_version:
        config["cuda_version"] = cuda_version
    
    # Check TensorFlow GPU
    config["tensorflow_gpu"] = check_tensorflow_gpu()
    
    # Configure GPU memory
    configure_gpu_memory()
    
    # Save configuration
    import json
    with open('gpu_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 GPU configuration saved to gpu_config.json")
    return config

def print_gpu_recommendations(config):
    """Print GPU optimization recommendations"""
    print("\n🚀 GPU Optimization Recommendations:")
    print("=" * 50)
    
    if config["gpu_available"]:
        print(f"🎮 GPU Type: {config['gpu_type']}")
        print(f"📦 Batch Size: {config['recommended_batch_size']}")
        print(f"🖼️  Image Size: {config['recommended_image_size']}x{config['recommended_image_size']}")
        print(f"⚡ Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
        
        if config['gpu_type'] == 'RTX_3050':
            print("\n💡 RTX 3050 Specific Tips:")
            print("  - Use mixed precision training for 2x speedup")
            print("  - Batch size 64 should work well with 8GB VRAM")
            print("  - Monitor GPU memory usage during training")
            
        elif config['gpu_type'] == 'GTX_1200':
            print("\n💡 GTX 1200 Series Tips:")
            print("  - Use batch size 32 to avoid VRAM issues")
            print("  - Mixed precision may not be available")
            print("  - Consider gradient accumulation for larger effective batch size")
        
        print(f"\n⚙️  To use GPU training, run:")
        print(f"  python train_modern.py --gpu --batch-size {config['recommended_batch_size']}")
        
    else:
        print("❌ No compatible GPU detected")
        print("💡 Recommendations for CPU training:")
        print("  - Use smaller batch size (16-32)")
        print("  - Consider using fewer epochs")
        print("  - Enable CPU optimizations")

def benchmark_gpu():
    """Run a quick GPU benchmark"""
    print("\n🏃 Running GPU benchmark...")
    
    try:
        import tensorflow as tf
        import time
        
        # Create test data
        test_data = tf.random.normal((32, 224, 224, 3))
        
        # Test GPU performance
        with tf.device('/GPU:0'):
            start_time = time.time()
            
            # Simple convolution operation
            conv = tf.keras.layers.Conv2D(64, 3, activation='relu')
            result = conv(test_data)
            
            # Force execution
            _ = result.numpy()
            
            gpu_time = time.time() - start_time
            print(f"✅ GPU benchmark: {gpu_time:.4f} seconds")
        
        # Test CPU performance for comparison
        with tf.device('/CPU:0'):
            start_time = time.time()
            
            conv_cpu = tf.keras.layers.Conv2D(64, 3, activation='relu')
            result_cpu = conv_cpu(test_data)
            
            _ = result_cpu.numpy()
            
            cpu_time = time.time() - start_time
            print(f"⚖️  CPU benchmark: {cpu_time:.4f} seconds")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main GPU setup function"""
    print("=" * 60)
    print("🚀 GPU Setup for Gender Detection")
    print("Supporting NVIDIA RTX 3050, GTX 1200 series, and more")
    print("=" * 60)
    
    # Create GPU configuration
    config = create_gpu_config()
    
    # Install GPU packages if needed
    if config["gpu_available"] and config["cuda_version"]:
        install_gpu_packages(config["cuda_version"])
    
    # Print recommendations
    print_gpu_recommendations(config)
    
    # Run benchmark if GPU is available
    if config["tensorflow_gpu"]:
        benchmark_gpu()
    
    print("\n" + "=" * 60)
    print("🎉 GPU setup complete!")
    print("💾 Configuration saved to gpu_config.json")
    print("📚 Check the recommendations above for optimal training")
    print("=" * 60)

if __name__ == "__main__":
    main()
