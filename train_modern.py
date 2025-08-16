#!/usr/bin/env python3
"""
Modern Gender Detection Training Script
Using ResNet50 backbone with advanced data augmentation and transfer learning
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
                        if 'RTX 3050' in gpu_name:
                            print("  🚀 RTX 3050 detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            self.use_mixed_precision = True
                        elif 'GTX 12' in gpu_name:
                            print("  🚀 GTX 1200 series detected - Standard precision")
                            self.use_mixed_precision = False
                        elif 'RTX' in gpu_name:
                            print("  🚀 RTX series detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            self.use_mixed_precision = True
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
                
                # Set GPU memory limit if needed (uncomment and adjust as needed)
                # tf.config.experimental.set_memory_limit(gpus[0], 6144)  # 6GB limit
                
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
        else:
            print("🖥️  Running on CPU")
            self.use_mixed_precision = False
            
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with advanced augmentation"""
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
        
    def create_augmentation_pipeline(self):
        """Create advanced data augmentation pipeline using Albumentations"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                p=0.3
            )
        ])
    
    def augment_data(self, X, y, augment_factor=2):
        """Apply data augmentation to increase dataset size"""
        print(f"\n🔄 Applying data augmentation (factor: {augment_factor})...")
        
        augmentation = self.create_augmentation_pipeline()
        
        augmented_images = []
        augmented_labels = []
        
        for _ in range(augment_factor):
            for img, label in tqdm(zip(X, y), total=len(X), desc="Augmenting"):
                # Convert to uint8 for albumentations
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
        """Create modern CNN model with ResNet50 backbone"""
        print("\n🏗️  Creating modern CNN model with ResNet50...")
        
        # Load pre-trained ResNet50 without top layers
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Build the complete model
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
            layers.Dense(self.num_classes, activation='softmax', dtype='float32')  # Keep output as float32 for mixed precision
        ])
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Wrap optimizer for mixed precision if enabled
        if self.use_mixed_precision:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            print("  ⚡ Mixed precision optimizer enabled")
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def create_callbacks(self):
        """Create training callbacks for better performance"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
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
            
            # CSV logger
            callbacks.CSVLogger('training_log.csv'),
            
            # TensorBoard
            callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train(self):
        """Complete training pipeline"""
        print("🎯 Starting training pipeline...")
        
        # Setup GPU
        self.setup_gpu()
        
        # Load data
        X, y = self.load_and_preprocess_data()
        
        # Apply data augmentation
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
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(
            y_val, y_pred_classes, 
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
        plt.show()
        print("💾 Training plots saved as 'training_history.png'")
    
    def save_config(self):
        """Save training configuration"""
        config = {
            'img_size': self.img_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'model_path': 'gender_detection_modern.keras'
        }
        
        with open('model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print("💾 Configuration saved as 'model_config.json'")

def main():
    """Main training function"""
    print("=" * 60)
    print("🚀 Modern Gender Detection Training")
    print("Using ResNet50 with Advanced Features")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModernGenderDetector(
        data_dir='gender_dataset_face',
        img_size=224,
        batch_size=32,
        epochs=50
    )
    
    # Start training
    try:
        model, history = trainer.train()
        print("\n🎉 Training completed successfully!")
        print("📁 Files generated:")
        print("  - gender_detection_modern.keras (main model)")
        print("  - best_gender_model.keras (best checkpoint)")
        print("  - model_config.json (configuration)")
        print("  - training_log.csv (training log)")
        print("  - training_history.png (plots)")
        print("  - logs/ (TensorBoard logs)")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
