#!/usr/bin/env python3
"""
Test script to verify the gender detection environment setup
"""

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        print(f"  OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
        print(f"  NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import cvlib as cv
        print("✓ cvlib imported successfully")
    except ImportError as e:
        print(f"✗ cvlib import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✓ TensorFlow imported successfully")
        print(f"  TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        print("  This is likely due to missing Microsoft Visual C++ Redistributable")
        print("  Please install it from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False
    
    try:
        from tensorflow.keras.preprocessing.image import img_to_array
        from tensorflow.keras.models import load_model
        print("✓ TensorFlow Keras components imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow Keras import failed: {e}")
        return False
    
    return True

def test_webcam():
    """Test webcam access"""
    print("\nTesting webcam access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Webcam access successful")
            cap.release()
            return True
        else:
            print("✗ Cannot access webcam")
            return False
    except Exception as e:
        print(f"✗ Webcam test failed: {e}")
        return False

def test_model():
    """Test if the gender detection model exists"""
    print("\nTesting model file...")
    import os
    model_path = 'gender_detection.model'
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        print(f"  Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("  You may need to train the model first by running: python train.py")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Gender Detection Environment Test")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test webcam (only if imports passed)
    if all_tests_passed:
        if not test_webcam():
            all_tests_passed = False
    
    # Test model
    if not test_model():
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED! Environment is ready.")
        print("You can now run: python detect_gender_webcam.py")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install Microsoft Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Ensure webcam is not being used by another application")
        print("3. Train the model if it doesn't exist: python train.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
