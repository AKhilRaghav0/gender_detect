#!/usr/bin/env python3
"""
GPU Package Installation Script
Automatically detects and installs appropriate packages for your GPU
"""

import subprocess
import sys
import platform
from pathlib import Path

def get_python_executable():
    """Get the correct Python executable path"""
    if Path('gender_detection_env/Scripts/python.exe').exists():
        return 'gender_detection_env/Scripts/python.exe'
    elif Path('gender_detection_env/bin/python').exists():
        return 'gender_detection_env/bin/python'
    else:
        return sys.executable

def check_nvidia_gpu():
    """Check for NVIDIA GPU"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Parse GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    if 'RTX 3050' in line:
                        return 'RTX_3050'
                    elif 'GTX 12' in line:
                        return 'GTX_1200'
                    elif 'RTX' in line:
                        return 'RTX_SERIES'
                    elif 'GTX' in line:
                        return 'GTX_SERIES'
            return 'NVIDIA_OTHER'
        return None
    except:
        return None

def check_cuda_version():
    """Check CUDA version"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    if 'V11.' in line:
                        return '11.x'
                    elif 'V12.' in line:
                        return '12.x'
        return None
    except:
        return None

def install_package(python_exe, package):
    """Install a package using pip"""
    try:
        print(f"  Installing {package}...")
        result = subprocess.run([python_exe, '-m', 'pip', 'install', package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {package} installed successfully")
            return True
        else:
            print(f"  ⚠️  {package} installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Error installing {package}: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("🚀 GPU Package Installer for Gender Detection")
    print("=" * 60)
    
    python_exe = get_python_executable()
    print(f"🐍 Using Python: {python_exe}")
    
    # Check GPU
    gpu_type = check_nvidia_gpu()
    cuda_version = check_cuda_version()
    
    print(f"🎮 GPU Type: {gpu_type or 'Not detected'}")
    print(f"🔧 CUDA Version: {cuda_version or 'Not detected'}")
    
    # Base packages (always install)
    base_packages = [
        'psutil',
        'memory-profiler',
    ]
    
    # GPU-specific packages
    gpu_packages = []
    
    if gpu_type:
        gpu_packages.extend([
            'pynvml',
            'gpustat'
        ])
        
        # CUDA-specific packages
        if cuda_version == '11.x':
            gpu_packages.append('cupy-cuda11x')
        elif cuda_version == '12.x':
            gpu_packages.append('cupy-cuda12x')
    
    # Install packages
    print(f"\n📦 Installing {len(base_packages + gpu_packages)} packages...")
    
    success_count = 0
    total_packages = len(base_packages + gpu_packages)
    
    for package in base_packages + gpu_packages:
        if install_package(python_exe, package):
            success_count += 1
    
    print(f"\n📊 Installation Results: {success_count}/{total_packages} packages installed")
    
    if success_count == total_packages:
        print("🎉 All packages installed successfully!")
    else:
        print("⚠️  Some packages failed to install")
    
    # GPU-specific recommendations
    if gpu_type == 'RTX_3050':
        print(f"\n💡 RTX 3050 Recommendations:")
        print("  - Your GPU supports mixed precision training")
        print("  - Recommended batch size: 64")
        print("  - Enable mixed precision with --mixed-precision flag")
        
    elif gpu_type == 'GTX_1200':
        print(f"\n💡 GTX 1200 Series Recommendations:")
        print("  - Your GPU may not support mixed precision")
        print("  - Recommended batch size: 32")
        print("  - Use standard precision training")
        
    elif gpu_type and 'RTX' in gpu_type:
        print(f"\n💡 RTX Series Recommendations:")
        print("  - Your GPU supports advanced features")
        print("  - Recommended batch size: 128+")
        print("  - Enable mixed precision for maximum performance")
        
    elif gpu_type and 'GTX' in gpu_type:
        print(f"\n💡 GTX Series Recommendations:")
        print("  - Your GPU provides good performance")
        print("  - Recommended batch size: 32-64")
        print("  - Mixed precision may or may not be supported")
    
    else:
        print(f"\n💡 CPU-only Recommendations:")
        print("  - Use smaller batch sizes (16-32)")
        print("  - Training will be slower but still functional")
        print("  - Consider using cloud GPU services for faster training")
    
    print(f"\n🚀 Next steps:")
    print("  1. Run: python gpu_setup.py (to configure GPU)")
    print("  2. Run: python benchmark_gpu.py (to test performance)")
    print("  3. Run: python train_modern.py (to start training)")
    print("  4. Run: python detect_gender_modern.py --gpu (for GPU-accelerated inference)")

if __name__ == "__main__":
    main()
