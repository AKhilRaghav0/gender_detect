#!/usr/bin/env python3
"""
Test script to debug model loading issues
"""

import os
# Set environment variable for TensorFlow 2.20.0 compatibility
os.environ['TF_USE_LEGACY_KERAS'] = 'true'

import tensorflow as tf
import numpy as np

def test_model_loading():
    """Test different model loading methods"""
    
    model_files = [
        'gender_detection.model',
        'gender_detection_modern.keras',
        'best_gender_model.keras'
    ]
    
    for model_file in model_files:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_file}")
        print(f"{'='*50}")
        
        if not os.path.exists(model_file):
            print(f"❌ File not found: {model_file}")
            continue
            
        try:
            # Method 1: Try tf.keras
            print("Method 1: tf.keras.models.load_model")
            model = tf.keras.models.load_model(model_file)
            print(f"✅ Successfully loaded with tf.keras: {model_file}")
            print(f"Model summary:")
            model.summary()
            return model
            
        except Exception as e1:
            print(f"❌ tf.keras failed: {e1}")
            
            try:
                # Method 2: Try tf.saved_model
                print("Method 2: tf.saved_model.load")
                model = tf.saved_model.load(model_file)
                print(f"✅ Successfully loaded with tf.saved_model: {model_file}")
                return model
                
            except Exception as e2:
                print(f"❌ tf.saved_model failed: {e2}")
                
                try:
                    # Method 3: Try direct h5py
                    print("Method 3: h5py direct loading")
                    import h5py
                    with h5py.File(model_file, 'r') as f:
                        print(f"✅ File can be opened with h5py: {model_file}")
                        print(f"Keys: {list(f.keys())}")
                        
                except Exception as e3:
                    print(f"❌ h5py failed: {e3}")
                    
                    try:
                        # Method 4: Try pickle
                        print("Method 4: pickle loading")
                        import pickle
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        print(f"✅ Successfully loaded with pickle: {model_file}")
                        return model
                        
                    except Exception as e4:
                        print(f"❌ pickle failed: {e4}")
    
    print("\n❌ All loading methods failed")
    return None

if __name__ == "__main__":
    print("🧪 Testing model loading methods...")
    model = test_model_loading()
    
    if model:
        print(f"\n✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test a simple prediction
        try:
            # Create a dummy input
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            prediction = model.predict(dummy_input, verbose=0)
            print(f"✅ Prediction successful! Output shape: {prediction.shape}")
            print(f"Prediction: {prediction}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    else:
        print("\n❌ Could not load any model")
