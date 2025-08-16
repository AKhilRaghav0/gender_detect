#!/usr/bin/env python3
"""
Test script for modern gender detection setup
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all modern packages can be imported"""
    print("🧪 Testing modern package imports...")
    
    packages = [
        ('tensorflow', 'TensorFlow'),
        ('tensorflow_hub', 'TensorFlow Hub'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('seaborn', 'Seaborn'),
        ('albumentations', 'Albumentations'),
        ('tqdm', 'TQDM'),
        ('pandas', 'Pandas')
    ]
    
    success_count = 0
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
    
    print(f"\n📊 Import Results: {success_count}/{len(packages)} packages imported successfully")
    return success_count == len(packages)

def test_dataset():
    """Test dataset structure and accessibility"""
    print("\n📁 Testing dataset structure...")
    
    data_dir = Path('gender_dataset_face')
    if not data_dir.exists():
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    classes = ['man', 'woman']
    total_images = 0
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            print(f"  ✅ {class_name}: {len(image_files)} images")
            total_images += len(image_files)
        else:
            print(f"  ❌ {class_name} directory not found")
            return False
    
    print(f"📊 Total images: {total_images}")
    return total_images > 0

def test_gpu_setup():
    """Test GPU configuration"""
    print("\n🖥️  Testing GPU setup...")
    
    try:
        import tensorflow as tf
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ {len(gpus)} GPU(s) detected:")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu}")
        else:
            print("  ℹ️  No GPU detected, using CPU")
        
        # Test basic TensorFlow operations
        with tf.device('/CPU:0'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = tf.add(a, b)
            print(f"  ✅ TensorFlow operations working: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ GPU/TensorFlow test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture creation"""
    print("\n🏗️  Testing model architecture...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        
        # Test ResNet50 creation
        model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        print(f"  ✅ ResNet50 backbone created: {model.count_params():,} parameters")
        
        # Test custom model creation
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        custom_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(f"  ✅ Custom model created: {custom_model.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model architecture test failed: {e}")
        return False

def test_data_augmentation():
    """Test data augmentation pipeline"""
    print("\n🔄 Testing data augmentation...")
    
    try:
        import albumentations as A
        import cv2
        
        # Create augmentation pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(p=0.3)
        ])
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        augmented = transform(image=dummy_image)
        
        print(f"  ✅ Augmentation pipeline working")
        print(f"  📐 Original shape: {dummy_image.shape}")
        print(f"  📐 Augmented shape: {augmented['image'].shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data augmentation test failed: {e}")
        return False

def check_training_progress():
    """Check if training is in progress"""
    print("\n🏃 Checking training progress...")
    
    # Check for training files
    training_files = [
        'training_log.csv',
        'logs',
        'best_gender_model.keras',
        'gender_detection_modern.keras'
    ]
    
    found_files = []
    for file in training_files:
        if Path(file).exists():
            found_files.append(file)
            print(f"  ✅ Found: {file}")
        else:
            print(f"  ⏳ Not found: {file}")
    
    if found_files:
        print(f"  📊 Training files found: {len(found_files)}/{len(training_files)}")
        
        # Check training log if exists
        if Path('training_log.csv').exists():
            try:
                import pandas as pd
                log = pd.read_csv('training_log.csv')
                print(f"  📈 Training epochs completed: {len(log)}")
                if len(log) > 0:
                    last_epoch = log.iloc[-1]
                    print(f"  📊 Latest metrics - Accuracy: {last_epoch.get('accuracy', 'N/A'):.4f}, Val Accuracy: {last_epoch.get('val_accuracy', 'N/A'):.4f}")
            except Exception as e:
                print(f"  ⚠️  Could not read training log: {e}")
    else:
        print("  ℹ️  No training files found - training may not have started yet")

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Modern Gender Detection Setup Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Structure", test_dataset),
        ("GPU Setup", test_gpu_setup),
        ("Model Architecture", test_model_architecture),
        ("Data Augmentation", test_data_augmentation),
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    # Check training progress
    check_training_progress()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 All tests passed! System is ready for training and inference.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
