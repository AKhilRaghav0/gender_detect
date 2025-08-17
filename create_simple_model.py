#!/usr/bin/env python3
"""
Create a simple, lightweight gender detection model for Raspberry Pi
"""

import os
# Set environment variable for TensorFlow 2.20.0 compatibility
os.environ['TF_USE_LEGACY_KERAS'] = 'true'

import tensorflow as tf
import numpy as np

def create_simple_model():
    """Create a simple CNN model for gender detection"""
    
    print("🏗️  Creating simple gender detection model...")
    
    # Create a lightweight model
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(224, 224, 3)),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Global pooling and dense layers
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
    
    print("✅ Model created successfully!")
    print("📊 Model summary:")
    model.summary()
    
    return model

def test_model(model):
    """Test the model with dummy data"""
    
    print("\n🧪 Testing model...")
    
    try:
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✅ Model works! Output shape: {prediction.shape}")
        print(f"Sample prediction: {prediction}")
        print(f"Predicted class: {'man' if prediction[0][0] > prediction[0][1] else 'woman'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def save_model(model, filename='gender_detection_simple.keras'):
    """Save the model"""
    
    try:
        model.save(filename)
        print(f"\n💾 Model saved as: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None

def main():
    """Main function"""
    
    print("🚀 Simple Gender Detection Model Creator")
    print("=" * 50)
    print("This model is designed to work on Raspberry Pi and other devices")
    print("with limited resources and older TensorFlow versions.")
    print()
    
    # Create model
    model = create_simple_model()
    
    # Test model
    if test_model(model):
        # Save model
        saved_file = save_model(model)
        
        if saved_file:
            print(f"\n🎉 Success! Model created and saved: {saved_file}")
            print("\n📋 Model features:")
            print("  - Lightweight CNN architecture")
            print("  - Compatible with older TensorFlow versions")
            print("  - Optimized for CPU inference")
            print("  - Suitable for Raspberry Pi deployment")
            print("\n⚠️  Note: This model has random weights and needs training")
            print("   You can train it using your existing dataset")
            
        else:
            print("\n❌ Failed to save model")
    else:
        print("\n❌ Model creation failed")

if __name__ == "__main__":
    main()
