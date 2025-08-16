#!/usr/bin/env python3
"""
GPU-Fixed Gender Detection Training Script
Forces GPU detection and usage for 4GB VRAM GPUs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import json

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force use GPU 0

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50

import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def force_gpu_setup():
    """Force GPU setup and detection"""
    print("🔍 Forcing GPU detection and setup...")
    
    # List all physical devices
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(f"  {device}")
    
    # Get GPU devices specifically
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPU devices found: {len(gpus)}")
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  Memory growth enabled for: {gpu}")
            
            # Set memory limit for 4GB GPU (3.4GB = 85% of 4GB)
            tf.config.experimental.set_memory_limit(gpus[0], 3400)
            print("  Memory limit set to 3.4GB for 4GB VRAM")
            
            # Enable mixed precision for RTX 3050
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("  ⚡ Mixed precision enabled")
                use_mixed_precision = True
            except Exception as e:
                print(f"  Mixed precision failed: {e}")
                use_mixed_precision = False
            
            # Test GPU with a simple operation
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print(f"  ✅ GPU test successful: {result.numpy()}")
            
            print("🚀 GPU setup completed successfully!")
            return True, use_mixed_precision
            
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            print("Falling back to CPU")
            return False, False
    else:
        print("❌ No GPU detected!")
        print("Make sure NVIDIA drivers and CUDA are installed")
        return False, False

class GPUGenderDetector:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,
                 batch_size=24,
                 epochs=40):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        print(f"🎮 GPU Gender Detection Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs: {self.epochs}")
        
        # Setup GPU
        self.gpu_available, self.use_mixed_precision = force_gpu_setup()
        
        if not self.gpu_available:
            print("⚠️  WARNING: No GPU available, training will be very slow!")
            response = input("Continue with CPU training? (y/n): ")
            if response.lower() != 'y':
                exit(1)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("\n📊 Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load images from each class directory
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                raise ValueError(f"Directory not found: {class_dir}")
                
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            print(f"📁 {class_name}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"Loading {class_name}"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"❌ Error loading {img_path}: {e}")
                    
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"✅ Dataset loaded: {X.shape[0]} images")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        return X, y
        
    def create_augmentation_pipeline(self):
        """Create data augmentation pipeline"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
        ])
    
    def augment_data(self, X, y, augment_factor=1):
        """Apply data augmentation"""
        if augment_factor <= 0:
            return X, y
            
        print(f"\n🔄 Applying data augmentation (factor: {augment_factor})...")
        
        augmentation = self.create_augmentation_pipeline()
        
        augmented_images = []
        augmented_labels = []
        
        for _ in range(augment_factor):
            for img, label in tqdm(zip(X, y), total=len(X), desc="Augmenting"):
                img_uint8 = (img * 255).astype(np.uint8)
                augmented = augmentation(image=img_uint8)
                aug_img = augmented['image'].astype(np.float32) / 255.0
                
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        # Combine original and augmented data
        X_final = np.concatenate([X, np.array(augmented_images)])
        y_final = np.concatenate([y, np.array(augmented_labels)])
        
        print(f"✅ Augmentation complete: {X_final.shape[0]} total images")
        return X_final, y_final
    
    def create_model(self):
        """Create CNN model with ResNet50 backbone"""
        print("\n🏗️  Creating CNN model with ResNet50...")
        
        # Force model creation on GPU if available
        device = '/GPU:0' if self.gpu_available else '/CPU:0'
        print(f"🎯 Creating model on: {device}")
        
        with tf.device(device):
            # Load pre-trained ResNet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
            
            # Freeze early layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            
            # Build model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax', dtype='float32')
            ])
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=0.0001)
        
        # Wrap optimizer for mixed precision if enabled
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("  ⚡ Mixed precision optimizer enabled")
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        print(f"🎯 Model device placement: {device}")
        
        return model
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'best_gender_model_gpu.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger('training_log_gpu.csv'),
        ]
        
        return callbacks_list
    
    def train(self):
        """Complete training pipeline"""
        print("🚀 Starting GPU training pipeline...")
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Light augmentation for 4GB VRAM
        X_aug, y_aug = self.augment_data(X, y, augment_factor=1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug, y_aug, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_aug
        )
        
        print(f"📊 Training set: {X_train.shape[0]} images")
        print(f"📊 Validation set: {X_val.shape[0]} images")
        
        # Create model
        self.model = self.create_model()
        
        # Print model summary
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        # Force training on GPU
        device = '/GPU:0' if self.gpu_available else '/CPU:0'
        print(f"\n🎯 Training on device: {device}")
        
        # Train model
        print(f"\n🏃 Starting training for {self.epochs} epochs...")
        with tf.device(device):
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callback_list,
                verbose=1
            )
        
        # Load best model
        self.model = keras.models.load_model('best_gender_model_gpu.keras')
        
        # Final evaluation
        print("\n📊 Final Evaluation:")
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Validation Accuracy: {val_acc:.4f}")
        
        # Generate predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=self.class_names))
        
        # Save final model
        self.model.save('gender_detection_gpu.keras')
        print("💾 Model saved as 'gender_detection_gpu.keras'")
        
        # Plot training history
        self.plot_training_history()
        
        return self.model, self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_gpu.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("💾 Training plots saved as 'training_history_gpu.png'")

def main():
    """Main training function"""
    print("=" * 60)
    print("🎮 GPU Gender Detection Training")
    print("Forcing GPU usage for 4GB VRAM optimization")
    print("=" * 60)
    
    # Initialize trainer
    trainer = GPUGenderDetector(
        data_dir='gender_dataset_face',
        img_size=224,
        batch_size=24,  # Safe for 4GB VRAM
        epochs=40       # Reasonable training time
    )
    
    # Start training
    try:
        model, history = trainer.train()
        print("\n🎉 GPU training completed successfully!")
        print("📁 Files generated:")
        print("  - gender_detection_gpu.keras (main model)")
        print("  - best_gender_model_gpu.keras (best checkpoint)")
        print("  - training_log_gpu.csv (training log)")
        print("  - training_history_gpu.png (plots)")
        
        # Test the model
        print("\n🧪 Testing model...")
        test_data = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model.predict(test_data, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        result = trainer.class_names[predicted_class]
        print(f"✅ Model test successful! Prediction: {result} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
