#!/usr/bin/env python3
"""
Simple GPU Test Script
Check if TensorFlow can detect and use your GPU
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import subprocess

def check_nvidia_driver():
    """Check NVIDIA driver"""
    print("🔍 Checking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver working")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"🎮 GPU: {line.strip()}")
                    break
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU detection"""
    print("\n🔍 Testing TensorFlow GPU detection...")
    
    print(f"📦 TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is built with TensorFlow
    print(f"🔧 CUDA built with TensorFlow: {tf.test.is_built_with_cuda()}")
    
    # List all devices
    print("\n📱 All devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")
    
    # List GPU devices specifically
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🎮 GPU devices: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"    Details: {details}")
            except:
                print("    Details: Not available")
        return True
    else:
        print("❌ No GPU devices found")
        return False

def test_gpu_operation():
    """Test a simple GPU operation"""
    print("\n🧪 Testing GPU operation...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU available for testing")
        return False
    
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Test GPU operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
            c = tf.matmul(a, b)
            
            print(f"✅ GPU operation successful!")
            print(f"   Result: {c.numpy()}")
            return True
            
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        return False

def test_memory_limit():
    """Test setting GPU memory limit"""
    print("\n💾 Testing GPU memory limit...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPU available")
        return False
    
    try:
        # Set memory limit to 3.4GB for 4GB GPU
        tf.config.experimental.set_memory_limit(gpus[0], 3400)
        print("✅ Memory limit set to 3.4GB")
        return True
    except Exception as e:
        print(f"❌ Memory limit failed: {e}")
        return False

def test_mixed_precision():
    """Test mixed precision"""
    print("\n⚡ Testing mixed precision...")
    
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        policy = tf.keras.mixed_precision.global_policy()
        print(f"✅ Mixed precision enabled: {policy.name}")
        
        # Reset to default
        tf.keras.mixed_precision.set_global_policy('float32')
        return True
    except Exception as e:
        print(f"❌ Mixed precision failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("🎮 GPU Test for Gender Detection")
    print("=" * 50)
    
    tests = [
        ("NVIDIA Driver", check_nvidia_driver),
        ("TensorFlow GPU", test_tensorflow_gpu),
        ("GPU Operation", test_gpu_operation),
        ("Memory Limit", test_memory_limit),
        ("Mixed Precision", test_mixed_precision)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU is ready for training")
        print("\n🚀 Ready to run:")
        print("   python train_gpu_fixed.py")
    elif passed >= 3:
        print("⚠️  Most tests passed, GPU should work")
        print("\n🚀 Try running:")
        print("   python train_gpu_fixed.py")
    else:
        print("❌ GPU setup has issues")
        print("\n💡 Troubleshooting:")
        print("1. Install NVIDIA drivers")
        print("2. Install CUDA toolkit")
        print("3. Reinstall TensorFlow: pip install tensorflow")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
