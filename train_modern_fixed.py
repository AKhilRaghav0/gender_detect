#!/usr/bin/env python3
"""
FIXED Modern Gender Detection Training Script
Resolved batch size mismatch for GH200 GPU
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ModernGenderDetector:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,  # ResNet50 optimal size
                 batch_size=32,
                 epochs=50):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        self.use_mixed_precision = False
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        print(f"🚀 Modern Gender Detection Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs: {self.epochs}")
        
    def setup_gpu(self):
        """Configure GPU settings for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for all GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
                
                # Get GPU details
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                        print(f"  GPU {i}: {gpu_name}")
                        
                        # Optimize for specific GPU types
                        if 'GH200' in gpu_name:
                            print("  🚀 GH200 detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            self.use_mixed_precision = True
                        elif 'RTX 4090' in gpu_name:
                            print("  🚀 RTX 4090 detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            self.use_mixed_precision = True
                        elif 'RTX 3090' in gpu_name:
                            print("  🚀 RTX 3090 detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            self.use_mixed_precision = True
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
                
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
        else:
            print("🖥️  Running on CPU")
            self.use_mixed_precision = False
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("📊 Loading and preprocessing data...")
        
        images = []
        labels = []
        
        # Load man images
        man_dir = self.data_dir / 'man'
        if man_dir.exists():
            man_files = list(man_dir.glob('*.jpg')) + list(man_dir.glob('*.jpeg')) + list(man_dir.glob('*.png'))
            print(f"📁 man: {len(man_files)} images")
            
            for img_path in tqdm(man_files, desc="Loading man"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(0)  # 0 for man
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Load woman images
        woman_dir = self.data_dir / 'woman'
        if woman_dir.exists():
            woman_files = list(woman_dir.glob('*.jpg')) + list(woman_dir.glob('*.jpeg')) + list(woman_dir.glob('*.png'))
            print(f"📁 woman: {len(woman_files)} images")
            
            for img_path in tqdm(woman_files, desc="Loading woman"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0
                        images.append(img)
                        labels.append(1)  # 1 for woman
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError("No images found in dataset directories")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"✅ Dataset loaded: {len(images)} images")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
        print(f"📊 Training set: {len(X_train)} images")
        print(f"📊 Validation set: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self):
        """Create the ResNet50-based model"""
        print("🏗️  Creating modern CNN model with ResNet50...")
        
        # Base ResNet50 model
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create the full model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', dtype='float32')
        ])
        
        # Compile model
        if self.use_mixed_precision:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        
        return model
    
    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                'best_gender_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks
    
    def train(self):
        """Train the model"""
        # Setup GPU
        self.setup_gpu()
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_and_preprocess_data()
        
        # Create model
        self.model = self.create_model()
        
        # Print model summary
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        # Train model
        print(f"\n🚀 Starting training for {self.epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        # Load best model
        self.model = keras.models.load_model('best_gender_model.keras')
        
        # Evaluate on validation set
        print("\n📊 Final Evaluation:")
        val_loss, val_acc, val_precision, val_recall = self.model.evaluate(
            X_val, y_val, verbose=0
        )
        
        print(f"✅ Validation Accuracy: {val_acc:.4f}")
        print(f"✅ Validation Precision: {val_precision:.4f}")
        print(f"✅ Validation Recall: {val_recall:.4f}")
        
        # Generate predictions for detailed metrics
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(
            y_val_classes, y_pred_classes, 
            target_names=self.class_names
        ))
        
        # Save final model
        self.model.save('gender_detection_modern.keras')
        print("💾 Model saved as 'gender_detection_modern.keras'")
        
        # Plot training history
        self.plot_training_history()
        
        # Save training configuration
        self.save_config()
        
        return self.model, self.history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("📊 Training history plot saved as 'training_history.png'")
        plt.show()
    
    def save_config(self):
        """Save model configuration"""
        config = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'use_mixed_precision': self.use_mixed_precision
        }
        
        with open('model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 Model configuration saved as 'model_config.json'")

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 Modern Gender Detection Training")
    print("Using ResNet50 with Advanced Features")
    print("=" * 60)
    
    # Create trainer
    trainer = ModernGenderDetector(
        data_dir='gender_dataset_face',
        img_size=224,
        batch_size=32,
        epochs=50
    )
    
    print("🎯 Starting training pipeline...")
    
    try:
        # Train model
        model, history = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Files saved:")
        print("   - gender_detection_modern.keras (trained model)")
        print("   - model_config.json (configuration)")
        print("   - training_history.png (training curves)")
        print("   - best_gender_model.keras (best checkpoint)")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
