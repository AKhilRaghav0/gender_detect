#!/usr/bin/env python3
"""
Optimized Training Script for 4GB VRAM GPUs
Specifically designed for RTX 3050 and similar GPUs
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def setup_gpu_for_4gb():
    """Setup GPU optimally for 4GB VRAM"""
    print("Setting up GPU for 4GB VRAM...")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit to 85% of 4GB (3.4GB)
            tf.config.experimental.set_memory_limit(gpus[0], 3400)
            print("GPU memory limit set to 3.4GB (85% of 4GB)")
            
            # Enable mixed precision for RTX 3050
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision enabled for 2x speedup")
            
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU detected, using CPU")
        return False

def main():
    """Main training function optimized for 4GB VRAM"""
    print("=" * 60)
    print("4GB VRAM Optimized Gender Detection Training")
    print("Designed for RTX 3050 and similar GPUs")
    print("=" * 60)
    
    # Setup GPU
    gpu_available = setup_gpu_for_4gb()
    
    # Import after GPU setup
    try:
        from train_modern import ModernGenderDetector
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Make sure train_modern.py exists in the current directory")
        return
    
    # Optimal settings for 4GB VRAM
    optimal_settings = {
        'batch_size': 24,        # Safe for 4GB VRAM
        'img_size': 224,         # Standard size
        'epochs': 40,            # Reasonable for quick training
    }
    
    print(f"Optimal settings for 4GB VRAM:")
    print(f"  Batch Size: {optimal_settings['batch_size']}")
    print(f"  Image Size: {optimal_settings['img_size']}x{optimal_settings['img_size']}")
    print(f"  Epochs: {optimal_settings['epochs']}")
    print(f"  Mixed Precision: {'Yes' if gpu_available else 'No'}")
    print(f"  Memory Limit: 3.4GB")
    
    # Initialize trainer
    trainer = ModernGenderDetector(
        data_dir='gender_dataset_face',
        img_size=optimal_settings['img_size'],
        batch_size=optimal_settings['batch_size'],
        epochs=optimal_settings['epochs']
    )
    
    # Start training
    try:
        print("\nStarting training...")
        model, history = trainer.train()
        
        print("\nTraining completed successfully!")
        print("Model saved as: gender_detection_modern.keras")
        print("Best model saved as: best_gender_model.keras")
        
        # Test the model
        print("\nTesting model...")
        test_inference(model)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_inference(model):
    """Quick inference test"""
    try:
        import numpy as np
        
        # Create dummy test data
        test_data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Test prediction
        prediction = model.predict(test_data, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        class_names = ['man', 'woman']
        result = class_names[predicted_class]
        
        print(f"Model test successful!")
        print(f"Test prediction: {result} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"Model test failed: {e}")

if __name__ == "__main__":
    main()
