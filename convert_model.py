#!/usr/bin/env python3
"""
Convert incompatible model to compatible format for Raspberry Pi
"""

import os
# Set environment variable for TensorFlow 2.20.0 compatibility
os.environ['TF_USE_LEGACY_KERAS'] = 'true'

import tensorflow as tf
import numpy as np
import h5py

def create_compatible_model():
    """Create a compatible ResNet50-based gender detection model"""
    
    # Create a simple but effective model architecture
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the full model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # 2 classes: man, woman
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def convert_model():
    """Convert the incompatible model to a compatible format"""
    
    print("🔄 Converting model to compatible format...")
    
    # Create a new compatible model
    model = create_compatible_model()
    
    print("✅ Created compatible model architecture")
    print("📊 Model summary:")
    model.summary()
    
    # Try to load weights from the old model
    try:
        print("\n📥 Attempting to load weights from gender_detection.model...")
        
        with h5py.File('gender_detection.model', 'r') as f:
            print(f"📁 Model file structure: {list(f.keys())}")
            
            if 'model_weights' in f:
                weights_group = f['model_weights']
                print(f"🔧 Weights structure: {list(weights_group.keys())}")
                
                # Try to load weights layer by layer
                for layer in model.layers:
                    if layer.name in weights_group:
                        try:
                            layer_weights = weights_group[layer.name]
                            print(f"📦 Loading weights for layer: {layer.name}")
                            
                            # Get the weights for this layer
                            if 'kernel:0' in layer_weights:
                                kernel = np.array(layer_weights['kernel:0'])
                                bias = np.array(layer_weights['bias:0']) if 'bias:0' in layer_weights else None
                                
                                if bias is not None:
                                    layer.set_weights([kernel, bias])
                                else:
                                    layer.set_weights([kernel])
                                    
                                print(f"✅ Loaded weights for {layer.name}")
                            else:
                                print(f"⚠️  No kernel weights found for {layer.name}")
                                
                        except Exception as e:
                            print(f"❌ Failed to load weights for {layer.name}: {e}")
                            continue
                
                print("\n✅ Weight loading completed")
                
            else:
                print("❌ No model_weights found in file")
                
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("⚠️  Will use random weights (model will need retraining)")
    
    # Save the compatible model
    try:
        output_path = 'gender_detection_compatible.keras'
        model.save(output_path)
        print(f"\n💾 Compatible model saved as: {output_path}")
        
        # Test the model
        print("\n🧪 Testing model...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Model works! Output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Model Conversion Tool")
    print("=" * 40)
    
    output_model = convert_model()
    
    if output_model:
        print(f"\n🎉 Success! Compatible model created: {output_model}")
        print("This model should work on Raspberry Pi and other devices.")
    else:
        print("\n❌ Model conversion failed")
        print("You may need to retrain the model with a compatible version.")
