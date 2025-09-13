#!/usr/bin/env python3
"""
Advanced Data Augmentation Training Script
Maximum data diversity and model robustness
Optimized for GH200 GPU with intensive augmentation
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
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

import albumentations as A
from albumentations.pytorch import ToTensorV2

class AdvancedAugmentationTrainer:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,
                 batch_size=32,
                 epochs=50,
                 augmentation_factor=3):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.augmentation_factor = augmentation_factor
        self.model = None
        self.history = None
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        print(f"🚀 Advanced Augmentation Gender Detection Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs: {self.epochs}")
        print(f"🎨 Augmentation factor: {self.augmentation_factor}x")
        
    def setup_gpu(self):
        """Configure GPU for maximum performance"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
                
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                        print(f"  GPU {i}: {gpu_name}")
                        
                        if 'GH200' in gpu_name:
                            print("  🚀 GH200 detected - Enabling mixed precision + XLA + Memory Growth")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            tf.config.optimizer.set_jit(True)
                            # Set memory limit to use full GH200 capacity
                            tf.config.experimental.set_memory_limit(gpus[0], 90000)  # 90GB
                        elif 'RTX' in gpu_name:
                            print("  🚀 RTX detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
                
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
        else:
            print("🖥️  Running on CPU")
    
    def create_advanced_augmentations(self):
        """Create intensive augmentation pipeline"""
        print("🎨 Creating advanced augmentation pipeline...")
        
        # Basic augmentations
        basic_aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        # Advanced augmentations
        advanced_aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
                A.ISONoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=5, p=0.2),
                A.Blur(blur_limit=5, p=0.2),
                A.GaussianBlur(blur_limit=5, p=0.2),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=60, p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.4),
                A.GridDistortion(p=0.2),
                A.IAAPiecewiseAffine(p=0.4),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=3),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.RandomGamma(),
            ], p=0.4),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.OneOf([
                A.RGBShift(p=0.3),
                A.ChannelShuffle(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.GridDropout(p=0.3),
            ], p=0.3),
        ])
        
        # Extreme augmentations for maximum robustness
        extreme_aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(var_limit=(10.0, 50.0)),
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.ISONoise(intensity=(0.1, 0.5)),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.4),
                A.MedianBlur(blur_limit=7, p=0.3),
                A.Blur(blur_limit=7, p=0.3),
                A.GaussianBlur(blur_limit=7, p=0.3),
            ], p=0.4),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.4, rotate_limit=90, p=0.4),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
                A.IAAPiecewiseAffine(scale=(0.01, 0.05), p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.4),
            ], p=0.4),
            A.OneOf([
                A.CLAHE(clip_limit=4),
                A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.1, 0.9)),
                A.IAAEmboss(alpha=(0.2, 0.5), strength=(0.2, 0.7)),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                A.RandomGamma(gamma_limit=(80, 120)),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.OneOf([
                A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.4),
                A.ChannelShuffle(p=0.4),
            ], p=0.4),
            A.OneOf([
                A.CoarseDropout(max_holes=12, max_height=48, max_width=48, p=0.4),
                A.GridDropout(ratio=0.3, p=0.4),
            ], p=0.4),
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.3),
            ], p=0.3),
        ])
        
        return [basic_aug, advanced_aug, extreme_aug]
    
    def load_and_augment_data(self):
        """Load data and apply intensive augmentation"""
        print("📊 Loading and augmenting dataset...")
        
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
                        labels.append(0)
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
                        labels.append(1)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        if not images:
            raise ValueError("No images found in dataset directories")
        
        print(f"✅ Original dataset loaded: {len(images)} images")
        print(f"📊 Class distribution: {np.bincount(labels)}")
        
        # Apply augmentation
        print(f"🎨 Applying {self.augmentation_factor}x augmentation...")
        augmentation_pipelines = self.create_advanced_augmentations()
        
        augmented_images = []
        augmented_labels = []
        
        for i, (img, label) in enumerate(tqdm(zip(images, labels), desc="Augmenting images", total=len(images))):
            # Add original image
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Apply different augmentation levels
            for aug_level in range(self.augmentation_factor - 1):
                if aug_level < len(augmentation_pipelines):
                    aug_pipeline = augmentation_pipelines[aug_level]
                else:
                    # Use random augmentation if we need more
                    aug_pipeline = random.choice(augmentation_pipelines)
                
                try:
                    # Convert to uint8 for albumentations
                    img_uint8 = (img * 255).astype(np.uint8)
                    augmented = aug_pipeline(image=img_uint8)['image']
                    # Convert back to float32
                    augmented = augmented.astype(np.float32) / 255.0
                    augmented_images.append(augmented)
                    augmented_labels.append(label)
                except Exception as e:
                    print(f"Augmentation failed for image {i}: {e}")
                    # Add original image as fallback
                    augmented_images.append(img)
                    augmented_labels.append(label)
        
        X = np.array(augmented_images)
        y = np.array(augmented_labels)
        
        print(f"✅ Augmentation complete: {len(X)} total images")
        print(f"📊 Final class distribution: {np.bincount(y)}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
        
        print(f"📊 Training set: {len(X_train)} images")
        print(f"📊 Validation set: {len(X_val)} images")
        
        return X_train, X_val, y_train, y_val
    
    def create_advanced_model(self):
        """Create advanced model architecture"""
        print("🏗️  Creating advanced CNN model...")
        
        # Use EfficientNetB0 as base for better performance
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Create advanced architecture
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            
            # Advanced dense layers with residual connections
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax', dtype='float32')
        ])
        
        # Compile with advanced optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        print(f"✅ Advanced model created with {model.count_params():,} parameters")
        return model
    
    def create_advanced_callbacks(self):
        """Create advanced training callbacks"""
        return [
            # Advanced early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Advanced learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.6,
                patience=6,
                min_lr=1e-8,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                'best_advanced_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs/advanced_training',
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ),
            
            # Custom callback for model unfreezing
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.unfreeze_layers(epoch)
            )
        ]
    
    def unfreeze_layers(self, epoch):
        """Unfreeze base model layers after certain epochs"""
        if epoch == 20 and self.model is not None:
            print("🔄 Unfreezing base model layers for fine-tuning...")
            base_model = self.model.layers[0]
            base_model.trainable = True
            
            # Recompile with lower learning rate
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=0.0001,  # Lower learning rate for fine-tuning
                weight_decay=0.01
            )
            
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall', 'auc']
            )
            print("✅ Base model layers unfrozen and recompiled")
    
    def train(self):
        """Train the advanced model"""
        # Setup GPU
        self.setup_gpu()
        
        # Load and augment data
        X_train, X_val, y_train, y_val = self.load_and_augment_data()
        
        # Create model
        self.model = self.create_advanced_model()
        
        # Print model summary
        print("\n📋 Advanced Model Architecture:")
        self.model.summary()
        
        # Create callbacks
        callback_list = self.create_advanced_callbacks()
        
        # Train model
        print(f"\n🚀 Starting advanced training for {self.epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        # Load best model
        self.model = keras.models.load_model('best_advanced_model.keras')
        
        # Evaluate
        print("\n📊 Final Evaluation:")
        val_loss, val_acc, val_precision, val_recall, val_auc = self.model.evaluate(
            X_val, y_val, verbose=0
        )
        
        print(f"✅ Validation Accuracy: {val_acc:.4f}")
        print(f"✅ Validation Precision: {val_precision:.4f}")
        print(f"✅ Validation Recall: {val_recall:.4f}")
        print(f"✅ Validation AUC: {val_auc:.4f}")
        
        # Save model
        self.model.save('gender_detection_advanced.keras')
        print("💾 Advanced model saved as 'gender_detection_advanced.keras'")
        
        # Plot results
        self.plot_training_results()
        
        return self.model, self.history
    
    def plot_training_results(self):
        """Plot advanced training results"""
        if self.history is None:
            return
            
        print("\n📈 Generating advanced training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Training Results', fontsize=16)
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(self.history.history['auc'], label='Training AUC')
        axes[1, 1].plot(self.history.history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('Model AUC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_training_results.png', dpi=300, bbox_inches='tight')
        print("📊 Advanced training results plot saved as 'advanced_training_results.png'")
        plt.show()

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 Advanced Data Augmentation Gender Detection Training")
    print("Maximum data diversity and model robustness")
    print("=" * 70)
    
    # Create trainer
    trainer = AdvancedAugmentationTrainer(
        data_dir='gender_dataset_face',
        img_size=224,
        batch_size=32,
        epochs=50,
        augmentation_factor=3
    )
    
    try:
        # Train model
        model, history = trainer.train()
        
        print("\n🎉 Advanced training completed successfully!")
        print("📁 Files saved:")
        print("   - gender_detection_advanced.keras (advanced model)")
        print("   - advanced_training_results.png (training curves)")
        print("   - best_advanced_model.keras (best checkpoint)")
        
    except Exception as e:
        print(f"\n❌ Advanced training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

