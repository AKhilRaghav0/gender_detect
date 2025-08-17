#!/usr/bin/env python3
"""
RTX 3050 Optimized Gender Detection Training
Optimized for RTX 3050 with 8GB VRAM and your dataset
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
import time

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

class RTX3050GenderTrainer:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,  # ResNet50 optimal size
                 batch_size=16,  # Optimized for RTX 3050
                 epochs=100):    # More epochs for better accuracy
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        print(f"🚀 RTX 3050 Optimized Gender Detection Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size} (optimized for RTX 3050)")
        print(f"🔄 Epochs: {self.epochs}")
        print(f"🎮 Target: RTX 3050 with 8GB VRAM")
        
    def setup_rtx3050(self):
        """Configure RTX 3050 for optimal performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth for RTX 3050
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
                
                # Get GPU details
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                        print(f"  GPU {i}: {gpu_name}")
                        
                        # RTX 3050 specific optimizations
                        if 'RTX 3050' in gpu_name:
                            print("  🚀 RTX 3050 detected - Applying optimizations:")
                            print("    - Enabling mixed precision for speed")
                            print("    - Optimizing batch size for 8GB VRAM")
                            print("    - Using memory-efficient training")
                            
                            # Enable mixed precision for RTX 3050
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            
                            # Set memory limit to 7GB (leave 1GB for system)
                            tf.config.experimental.set_memory_limit(gpu, 7168)
                            
                        elif 'RTX' in gpu_name:
                            print("  🚀 RTX series detected - Good performance expected")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
                
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
                print("🖥️  Falling back to CPU")
        else:
            print("🖥️  No GPU detected, using CPU")
            print("⚠️  Training will be much slower on CPU")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with RTX 3050 optimizations"""
        print("\n📊 Loading and preprocessing data...")
        
        # Initialize lists for data
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
                    # Load and preprocess image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img.astype(np.float32) / 255.0  # Normalize
                        
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
        
    def create_rtx3050_augmentation(self):
        """Create RTX 3050 optimized augmentation pipeline"""
        return A.Compose([
            # Basic augmentations (fast on RTX 3050)
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, 
                contrast_limit=0.15, 
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=0.5
            ),
            
            # RTX 3050 optimized augmentations
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            
            # Geometric augmentations (careful with RTX 3050)
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.08,
                rotate_limit=10,
                p=0.5
            ),
            
            # Advanced augmentations (RTX 3050 can handle these)
            A.CoarseDropout(
                max_holes=6,
                max_height=6,
                max_width=6,
                p=0.3
            ),
            
            # Color augmentations
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.RandomShadow(p=0.2),
            A.RandomSunFlare(p=0.1)
        ])
    
    def augment_data_rtx3050(self, X, y, augment_factor=3):
        """Apply RTX 3050 optimized data augmentation"""
        print(f"\n🔄 Applying RTX 3050 optimized augmentation (factor: {augment_factor})...")
        
        augmentation = self.create_rtx3050_augmentation()
        
        augmented_images = []
        augmented_labels = []
        
        # Process in smaller batches for RTX 3050
        batch_size = 100
        total_batches = (len(X) + batch_size - 1) // batch_size
        
        for _ in range(augment_factor):
            for batch_start in tqdm(range(0, len(X), batch_size), total=total_batches, desc="Augmenting"):
                batch_end = min(batch_start + batch_size, len(X))
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                
                for img, label in zip(batch_X, batch_y):
                    # Convert to uint8 for albumentations
                    img_uint8 = (img * 255).astype(np.uint8)
                    augmented = augmentation(image=img_uint8)
                    aug_img = augmented['image'].astype(np.float32) / 255.0
                    
                    augmented_images.append(aug_img)
                    augmented_labels.append(label)
        
        # Combine original and augmented data
        X_final = np.concatenate([X, np.array(augmented_images)])
        y_final = np.concatenate([y, np.array(augmented_labels)])
        
        print(f"✅ RTX 3050 augmentation complete: {X_final.shape[0]} total images")
        return X_final, y_final
    
    def create_rtx3050_model(self):
        """Create RTX 3050 optimized model"""
        print("\n🏗️  Creating RTX 3050 optimized model...")
        
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # RTX 3050 specific layer freezing strategy
        # Freeze more layers initially, then gradually unfreeze
        for layer in base_model.layers[:-30]:  # Freeze more layers for RTX 3050
            layer.trainable = False
        
        print(f"  🔒 Frozen {len(base_model.layers[:-30])} base layers")
        print(f"  🔓 Training {len(base_model.layers[-30:])} base layers")
        
        # Build the complete model optimized for RTX 3050
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.6),  # Higher dropout for RTX 3050
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),  # Additional layer for RTX 3050
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # RTX 3050 optimized optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.0001,  # Slightly lower for RTX 3050
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ RTX 3050 model created with {model.count_params():,} parameters")
        return model
    
    def create_rtx3050_callbacks(self):
        """Create RTX 3050 optimized training callbacks"""
        callbacks_list = [
            # Early stopping (more patient for RTX 3050)
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # More patience for RTX 3050
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction (RTX 3050 specific)
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,  # Gentler reduction for RTX 3050
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            callbacks.ModelCheckpoint(
                'rtx3050_best_gender_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # CSV logger
            callbacks.CSVLogger('rtx3050_training_log.csv'),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir='rtx3050_logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_rtx3050(self):
        """Complete RTX 3050 training pipeline"""
        print("🎯 Starting RTX 3050 training pipeline...")
        
        # Setup RTX 3050
        self.setup_rtx3050()
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Training set: {X_train.shape[0]} images")
        print(f"📊 Validation set: {X_val.shape[0]} images")
        
        # Apply RTX 3050 optimized augmentation
        X_train_aug, y_train_aug = self.augment_data_rtx3050(X_train, y_train)
        
        print(f"📊 After augmentation - Training set: {X_train_aug.shape[0]} images")
        
        # Create RTX 3050 model
        self.model = self.create_rtx3050_model()
        
        # Create callbacks
        callbacks_list = self.create_rtx3050_callbacks()
        
        # Train the model
        print("\n🚀 Starting training on RTX 3050...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Training completed in {training_time/60:.1f} minutes")
        
        # Evaluate final model
        self.evaluate_model(X_val, y_val)
        
        # Save final model
        self.save_model()
        
        return self.history
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate the trained model"""
        print("\n📊 Evaluating RTX 3050 trained model...")
        
        # Make predictions
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred_classes == y_val)
        
        print(f"✅ Final Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_val, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        print(f"\n📊 Confusion Matrix:")
        print(cm)
        
        return accuracy
    
    def save_model(self):
        """Save the trained model"""
        model_path = 'rtx3050_gender_detection_final.keras'
        self.model.save(model_path)
        print(f"\n💾 Final model saved as: {model_path}")
        
        # Save model info
        model_info = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware': 'RTX 3050 Optimized'
        }
        
        with open('rtx3050_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 Model info saved as: rtx3050_model_info.json")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
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
        plt.savefig('rtx3050_training_history.png', dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved as: rtx3050_training_history.png")

def main():
    """Main training function"""
    print("=" * 80)
    print("🚀 RTX 3050 Optimized Gender Detection Training")
    print("=" * 80)
    
    # Initialize trainer
    trainer = RTX3050GenderTrainer()
    
    try:
        # Start training
        history = trainer.train_rtx3050()
        
        # Plot results
        trainer.plot_training_history()
        
        print("\n🎉 RTX 3050 training completed successfully!")
        print("📁 Check the following files:")
        print("  - rtx3050_gender_detection_final.keras (trained model)")
        print("  - rtx3050_best_gender_model.keras (best checkpoint)")
        print("  - rtx3050_training_history.png (training plots)")
        print("  - rtx3050_training_log.csv (training logs)")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
