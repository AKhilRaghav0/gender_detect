#!/usr/bin/env python3
"""
Automatic Setup and Training Script
Detects GPU, installs packages, optimizes settings, and starts training
Works with any NVIDIA GPU: RTX 3050 (4GB), G200 server, GTX series, etc.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

class AutoGenderDetectionSetup:
    def __init__(self):
        self.gpu_info = None
        self.vram_gb = None
        self.cuda_version = None
        self.optimal_settings = {}
        
    def detect_system(self):
        """Detect GPU, VRAM, and system capabilities"""
        print("🔍 Detecting system configuration...")
        
        # Check NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        gpu_name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        self.vram_gb = vram_mb / 1024
                        
                        print(f"🎮 GPU Detected: {gpu_name}")
                        print(f"💾 VRAM: {self.vram_gb:.1f} GB")
                        
                        self.gpu_info = {
                            'name': gpu_name,
                            'vram_gb': self.vram_gb,
                            'type': self._classify_gpu(gpu_name)
                        }
                        break
            else:
                print("❌ No NVIDIA GPU detected")
                self.gpu_info = None
                
        except Exception as e:
            print(f"❌ GPU detection failed: {e}")
            self.gpu_info = None
        
        # Check CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        if 'V11.' in line:
                            self.cuda_version = '11.x'
                        elif 'V12.' in line:
                            self.cuda_version = '12.x'
                        print(f"🔧 CUDA Version: {self.cuda_version}")
                        break
        except:
            print("⚠️  CUDA not detected")
        
        return self.gpu_info is not None
    
    def _classify_gpu(self, gpu_name):
        """Classify GPU type for optimization"""
        gpu_name_upper = gpu_name.upper()
        
        if 'RTX 3050' in gpu_name_upper:
            return 'RTX_3050'
        elif 'RTX 4090' in gpu_name_upper or 'RTX 4080' in gpu_name_upper:
            return 'RTX_40_HIGH'
        elif 'RTX 30' in gpu_name_upper or 'RTX 40' in gpu_name_upper:
            return 'RTX_MODERN'
        elif 'RTX 20' in gpu_name_upper:
            return 'RTX_20_SERIES'
        elif 'GTX 16' in gpu_name_upper or 'GTX 12' in gpu_name_upper:
            return 'GTX_16_12_SERIES'
        elif 'GTX' in gpu_name_upper:
            return 'GTX_LEGACY'
        elif 'TESLA' in gpu_name_upper or 'V100' in gpu_name_upper or 'A100' in gpu_name_upper:
            return 'SERVER_GPU'
        elif 'T4' in gpu_name_upper or 'P100' in gpu_name_upper:
            return 'CLOUD_GPU'
        elif 'QUADRO' in gpu_name_upper:
            return 'WORKSTATION'
        else:
            return 'UNKNOWN'
    
    def calculate_optimal_settings(self):
        """Calculate optimal training settings based on GPU"""
        if not self.gpu_info:
            # CPU fallback
            self.optimal_settings = {
                'batch_size': 16,
                'image_size': 224,
                'epochs': 30,
                'mixed_precision': False,
                'memory_limit_mb': None,
                'gradient_accumulation': 2,
                'device': 'CPU'
            }
            print("🖥️  CPU-only settings configured")
            return
        
        gpu_type = self.gpu_info['type']
        vram_gb = self.gpu_info['vram_gb']
        
        # Base settings
        settings = {
            'device': 'GPU',
            'memory_limit_mb': None,
            'gradient_accumulation': 1
        }
        
        # VRAM-based optimization
        if vram_gb >= 24:  # High-end server GPUs
            settings.update({
                'batch_size': 128,
                'image_size': 256,
                'epochs': 50,
                'mixed_precision': True
            })
            print(f"🚀 High-end GPU settings (24+ GB VRAM)")
            
        elif vram_gb >= 12:  # RTX 4090, RTX 3080 Ti, etc.
            settings.update({
                'batch_size': 64,
                'image_size': 224,
                'epochs': 50,
                'mixed_precision': True
            })
            print(f"🚀 High-performance GPU settings (12+ GB VRAM)")
            
        elif vram_gb >= 8:  # RTX 3070, RTX 4060 Ti, etc.
            settings.update({
                'batch_size': 48,
                'image_size': 224,
                'epochs': 50,
                'mixed_precision': True
            })
            print(f"🚀 Mid-range GPU settings (8+ GB VRAM)")
            
        elif vram_gb >= 6:  # RTX 3060, GTX 1660 Ti, etc.
            settings.update({
                'batch_size': 32,
                'image_size': 224,
                'epochs': 50,
                'mixed_precision': True,
                'memory_limit_mb': int(vram_gb * 1024 * 0.9)  # Use 90% of VRAM
            })
            print(f"🎯 Mid-range GPU settings (6+ GB VRAM)")
            
        elif vram_gb >= 4:  # RTX 3050, GTX 1650, your current GPU
            settings.update({
                'batch_size': 24,
                'image_size': 224,
                'epochs': 40,
                'mixed_precision': True,
                'memory_limit_mb': int(vram_gb * 1024 * 0.85),  # Use 85% of VRAM
                'gradient_accumulation': 2  # Effective batch size 48
            })
            print(f"🎯 4GB VRAM optimized settings")
            
        else:  # Low VRAM GPUs
            settings.update({
                'batch_size': 16,
                'image_size': 224,
                'epochs': 30,
                'mixed_precision': False,
                'memory_limit_mb': int(vram_gb * 1024 * 0.8),
                'gradient_accumulation': 3
            })
            print(f"⚠️  Low VRAM GPU settings ({vram_gb:.1f} GB)")
        
        # GPU-specific adjustments
        if gpu_type in ['GTX_LEGACY', 'GTX_16_12_SERIES']:
            settings['mixed_precision'] = False  # Older GPUs may not support it
            print("  📝 Mixed precision disabled for older GPU")
        
        elif gpu_type == 'SERVER_GPU':
            settings['batch_size'] = min(settings['batch_size'] * 2, 256)  # Server GPUs can handle more
            print("  🏢 Server GPU optimization applied")
        
        self.optimal_settings = settings
        
        # Print summary
        print(f"\n📊 Optimal Settings:")
        print(f"  Batch Size: {settings['batch_size']}")
        print(f"  Image Size: {settings['image_size']}x{settings['image_size']}")
        print(f"  Epochs: {settings['epochs']}")
        print(f"  Mixed Precision: {'Yes' if settings['mixed_precision'] else 'No'}")
        print(f"  Memory Limit: {settings['memory_limit_mb']}MB" if settings['memory_limit_mb'] else "  Memory Limit: None")
        print(f"  Gradient Accumulation: {settings['gradient_accumulation']}")
        
        if settings['gradient_accumulation'] > 1:
            effective_batch = settings['batch_size'] * settings['gradient_accumulation']
            print(f"  Effective Batch Size: {effective_batch}")
    
    def install_packages(self):
        """Install required packages"""
        print("\n📦 Installing packages...")
        
        python_exe = self._get_python_executable()
        
        # Base packages
        base_packages = [
            'tensorflow>=2.15.0',
            'opencv-python>=4.8.0',
            'opencv-contrib-python>=4.8.0',
            'numpy>=1.24.0',
            'Pillow>=10.0.0',
            'matplotlib>=3.7.0',
            'scikit-learn>=1.3.0',
            'seaborn>=0.12.0',
            'albumentations>=1.3.0',
            'tqdm>=4.65.0',
            'pandas>=2.0.0',
            'psutil>=5.9.0',
            'memory-profiler>=0.60.0'
        ]
        
        # GPU-specific packages
        if self.gpu_info:
            base_packages.extend([
                'pynvml>=11.4.0',
                'gpustat>=1.0.0'
            ])
            
            # CUDA-specific packages
            if self.cuda_version == '11.x':
                base_packages.append('cupy-cuda11x>=12.0.0')
            elif self.cuda_version == '12.x':
                base_packages.append('cupy-cuda12x>=12.0.0')
        
        # Install packages
        success_count = 0
        for package in base_packages:
            try:
                print(f"  Installing {package}...")
                result = subprocess.run([python_exe, '-m', 'pip', 'install', package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    success_count += 1
                    print(f"  ✅ {package.split('>=')[0]} installed")
                else:
                    print(f"  ⚠️  {package.split('>=')[0]} failed")
            except Exception as e:
                print(f"  ❌ Error installing {package}: {e}")
        
        print(f"\n📊 Package Installation: {success_count}/{len(base_packages)} successful")
        return success_count > len(base_packages) * 0.8  # 80% success rate
    
    def _get_python_executable(self):
        """Get the correct Python executable"""
        if Path('gender_detection_env/Scripts/python.exe').exists():
            return 'gender_detection_env/Scripts/python.exe'
        elif Path('gender_detection_env/bin/python').exists():
            return 'gender_detection_env/bin/python'
        else:
            return sys.executable
    
    def create_optimized_training_script(self):
        """Create training script with optimal settings"""
        print("\n🏗️  Creating optimized training script...")
        
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated optimized training script
GPU: {self.gpu_info['name'] if self.gpu_info else 'CPU'}
VRAM: {self.vram_gb:.1f} GB
Settings: {json.dumps(self.optimal_settings, indent=2)}
"""

import os
import tensorflow as tf
from train_modern import ModernGenderDetector

def main():
    print("🚀 Starting optimized training...")
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            memory_limit = {self.optimal_settings.get('memory_limit_mb')}
            if memory_limit:
                tf.config.experimental.set_memory_limit(gpus[0], memory_limit)
                print(f"🎯 GPU memory limit set to {{memory_limit}}MB")
            
            # Configure mixed precision
            if {str(self.optimal_settings.get('mixed_precision', False)).lower()}:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("⚡ Mixed precision enabled")
        
        except RuntimeError as e:
            print(f"GPU configuration error: {{e}}")
    
    # Initialize trainer with optimal settings
    trainer = ModernGenderDetector(
        data_dir='gender_dataset_face',
        img_size={self.optimal_settings.get('image_size', 224)},
        batch_size={self.optimal_settings.get('batch_size', 32)},
        epochs={self.optimal_settings.get('epochs', 50)}
    )
    
    # Start training
    try:
        model, history = trainer.train()
        print("🎉 Training completed successfully!")
        
        # Save optimal settings
        import json
        with open('optimal_settings.json', 'w') as f:
            json.dump({json.dumps(self.optimal_settings, indent=2)}, f)
        
    except Exception as e:
        print(f"❌ Training failed: {{e}}")
        raise

if __name__ == "__main__":
    main()
'''
        
        with open('train_optimized.py', 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print("✅ Optimized training script created: train_optimized.py")
    
    def save_configuration(self):
        """Save system configuration"""
        config = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_info': self.gpu_info,
            'cuda_version': self.cuda_version,
            'optimal_settings': self.optimal_settings,
            'vram_gb': self.vram_gb
        }
        
        with open('system_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 Configuration saved to system_config.json")
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("=" * 60)
        print("🚀 Auto Gender Detection Setup")
        print("Universal GPU Support: RTX 3050, G200, GTX series, etc.")
        print("=" * 60)
        
        # Step 1: Detect system
        has_gpu = self.detect_system()
        
        # Step 2: Calculate optimal settings
        self.calculate_optimal_settings()
        
        # Step 3: Install packages
        if not self.install_packages():
            print("❌ Package installation failed")
            return False
        
        # Step 4: Create optimized scripts
        self.create_optimized_training_script()
        
        # Step 5: Save configuration
        self.save_configuration()
        
        # Step 6: Print summary and commands
        self.print_summary()
        
        return True
    
    def print_summary(self):
        """Print setup summary and next steps"""
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        if self.gpu_info:
            print(f"🎮 GPU: {self.gpu_info['name']}")
            print(f"💾 VRAM: {self.vram_gb:.1f} GB")
            print(f"⚡ Mixed Precision: {'Yes' if self.optimal_settings['mixed_precision'] else 'No'}")
            print(f"📦 Batch Size: {self.optimal_settings['batch_size']}")
            
            if self.optimal_settings['gradient_accumulation'] > 1:
                effective = self.optimal_settings['batch_size'] * self.optimal_settings['gradient_accumulation']
                print(f"📈 Effective Batch Size: {effective}")
        else:
            print("🖥️  CPU-only configuration")
        
        print(f"\n🚀 SINGLE-LINE COMMANDS:")
        print("=" * 30)
        
        # Training command
        python_exe = self._get_python_executable()
        print(f"📚 Start Training:")
        print(f"   {python_exe} train_optimized.py")
        
        print(f"\n🎥 Start Webcam Detection:")
        print(f"   {python_exe} detect_gender_modern.py --mode webcam --gpu")
        
        print(f"\n🖼️  Process Single Image:")
        print(f"   {python_exe} detect_gender_modern.py --mode image --image path/to/image.jpg --gpu")
        
        print(f"\n📊 Run Benchmark:")
        print(f"   {python_exe} benchmark_gpu.py")
        
        print(f"\n🔧 Monitor GPU:")
        print("   gpustat -i 1")
        print("   nvidia-smi -l 1")
        
        print("\n" + "=" * 60)
        print("📋 For G200 Server Deployment:")
        print("   1. Copy this folder to your G200 server")
        print("   2. Run: python auto_setup.py")
        print("   3. Run: python train_optimized.py")
        print("=" * 60)

def main():
    """Main setup function"""
    setup = AutoGenderDetectionSetup()
    success = setup.run_full_setup()
    
    if success:
        # Ask user if they want to start training immediately
        response = input("\n🚀 Start training now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🏃 Starting training...")
            python_exe = setup._get_python_executable()
            subprocess.run([python_exe, 'train_optimized.py'])
    else:
        print("❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()
