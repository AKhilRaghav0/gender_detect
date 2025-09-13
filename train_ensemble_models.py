#!/usr/bin/env python3
"""
Multi-Model Ensemble Training Script
Train multiple architectures and combine for maximum performance
Optimized for GH200 GPU
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
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, DenseNet121, 
    InceptionV3, Xception, MobileNetV2
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

class EnsembleModelTrainer:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,
                 batch_size=32,
                 epochs=40):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}
        self.histories = {}
        self.model_scores = {}
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        # Model configurations
        self.model_configs = {
            'resnet50': {
                'base_model': ResNet50,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'ResNet50 - Deep Residual Network'
            },
            'efficientnet': {
                'base_model': EfficientNetB0,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'EfficientNetB0 - Efficient Architecture'
            },
            'densenet': {
                'base_model': DenseNet121,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'DenseNet121 - Dense Connections'
            },
            'inception': {
                'base_model': InceptionV3,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'InceptionV3 - Multi-Scale Processing'
            },
            'xception': {
                'base_model': Xception,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'Xception - Depthwise Separable Convolutions'
            },
            'mobilenet': {
                'base_model': MobileNetV2,
                'weights': 'imagenet',
                'input_shape': (self.img_size, self.img_size, 3),
                'description': 'MobileNetV2 - Lightweight Architecture'
            }
        }
        
        print(f"🚀 Multi-Model Ensemble Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs per model: {self.epochs}")
        print(f"🏗️  Models to train: {len(self.model_configs)}")
        
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
                            print("  🚀 GH200 detected - Enabling mixed precision + XLA + Full Memory")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            tf.config.optimizer.set_jit(True)
                            # Use full GH200 capacity
                            tf.config.experimental.set_memory_limit(gpus[0], 95000)  # 95GB
                        elif 'RTX' in gpu_name:
                            print("  🚀 RTX detected - Enabling mixed precision")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
                
            except RuntimeError as e:
                print(f"❌ GPU configuration error: {e}")
        else:
            print("🖥️  Running on CPU")
    
    def load_data(self):
        """Load dataset"""
        print("📊 Loading dataset...")
        
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
        
        X = np.array(images)
        y = np.array(labels)
        
        print(f"✅ Dataset loaded: {len(images)} images")
        print(f"📊 Class distribution: {np.bincount(y)}")
        
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
    
    def create_model(self, model_name, config):
        """Create a specific model architecture"""
        print(f"🏗️  Creating {model_name} model...")
        
        try:
            # Create base model
            base_model = config['base_model'](
                weights=config['weights'],
                include_top=False,
                input_shape=config['input_shape']
            )
            
            # Freeze base model initially
            base_model.trainable = False
            
            # Create full model with architecture-specific layers
            if model_name == 'resnet50':
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
                    layers.Dropout(0.2),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            elif model_name == 'efficientnet':
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.6),
                    layers.Dense(1024, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            elif model_name == 'densenet':
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
                    layers.Dropout(0.2),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            elif model_name == 'inception':
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.6),
                    layers.Dense(1024, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            elif model_name == 'xception':
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
                    layers.Dropout(0.2),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            elif model_name == 'mobilenet':
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(256, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.2),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            else:
                # Default architecture
                model = keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(self.num_classes, activation='softmax', dtype='float32')
                ])
            
            # Compile model
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=0.001,
                weight_decay=0.01
            )
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            print(f"✅ {model_name} model created with {model.count_params():,} parameters")
            return model
            
        except Exception as e:
            print(f"❌ Failed to create {model_name} model: {e}")
            return None
    
    def create_callbacks(self, model_name):
        """Create callbacks for a specific model"""
        return [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                f'best_{model_name}_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{model_name}',
                histogram_freq=1,
                write_graph=True
            )
        ]
    
    def train_model(self, model_name, config, X_train, X_val, y_train, y_val):
        """Train a single model"""
        print(f"\n🎯 Training {model_name.upper()} model...")
        print(f"📝 Description: {config['description']}")
        
        start_time = time.time()
        
        # Create model
        model = self.create_model(model_name, config)
        if model is None:
            return None, None
        
        # Create callbacks
        callback_list = self.create_callbacks(model_name)
        
        # Train model
        print(f"🚀 Starting training for {self.epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        # Load best model
        model = keras.models.load_model(f'best_{model_name}_model.keras')
        
        # Evaluate
        val_loss, val_acc, val_precision, val_recall = model.evaluate(
            X_val, y_val, verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Store results
        self.model_scores[model_name] = {
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'loss': val_loss,
            'training_time': training_time,
            'parameters': model.count_params()
        }
        
        print(f"✅ {model_name.upper()} training completed:")
        print(f"   Accuracy: {val_acc:.4f}")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall: {val_recall:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")
        
        return model, history
    
    def train_all_models(self):
        """Train all models in parallel"""
        # Setup GPU
        self.setup_gpu()
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data()
        
        print(f"\n🔄 Starting training for {len(self.model_configs)} models...")
        
        # Train each model
        for model_name, config in self.model_configs.items():
            try:
                model, history = self.train_model(
                    model_name, config, X_train, X_val, y_train, y_val
                )
                
                if model is not None:
                    self.models[model_name] = model
                    self.histories[model_name] = history
                    
                    # Save individual model
                    model.save(f'gender_detection_{model_name}.keras')
                    print(f"💾 {model_name} model saved")
                else:
                    print(f"❌ {model_name} training failed, skipping...")
                    
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                continue
        
        # Calculate ensemble results
        if len(self.models) > 1:
            self.calculate_ensemble_results(X_val, y_val)
            self.save_ensemble_results()
            self.plot_ensemble_results()
        
        return self.models, self.model_scores
    
    def calculate_ensemble_results(self, X_val, y_val):
        """Calculate ensemble performance"""
        print("\n📊 Calculating ensemble results...")
        
        all_predictions = []
        model_names = list(self.models.keys())
        
        # Get predictions from all models
        for model_name in model_names:
            model = self.models[model_name]
            predictions = model.predict(X_val)
            all_predictions.append(predictions)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(all_predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        y_val_classes = np.argmax(y_val, axis=1)
        
        # Calculate ensemble metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_accuracy = accuracy_score(y_val_classes, ensemble_pred_classes)
        ensemble_precision = precision_score(y_val_classes, ensemble_pred_classes, average='weighted')
        ensemble_recall = recall_score(y_val_classes, ensemble_pred_classes, average='weighted')
        ensemble_f1 = f1_score(y_val_classes, ensemble_pred_classes, average='weighted')
        
        print(f"🎯 Ensemble Performance:")
        print(f"   Accuracy: {ensemble_accuracy:.4f}")
        print(f"   Precision: {ensemble_precision:.4f}")
        print(f"   Recall: {ensemble_recall:.4f}")
        print(f"   F1-Score: {ensemble_f1:.4f}")
        
        # Store ensemble results
        self.ensemble_results = {
            'accuracy': ensemble_accuracy,
            'precision': ensemble_precision,
            'recall': ensemble_recall,
            'f1_score': ensemble_f1,
            'models_used': model_names
        }
    
    def save_ensemble_results(self):
        """Save ensemble results and models"""
        print("\n💾 Saving ensemble results...")
        
        # Save ensemble results
        results = {
            'ensemble_results': self.ensemble_results,
            'individual_model_scores': self.model_scores,
            'training_params': {
                'img_size': self.img_size,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        
        with open('ensemble_training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Ensemble results saved!")
    
    def plot_ensemble_results(self):
        """Plot ensemble training results"""
        print("\n📈 Generating ensemble results plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Model Ensemble Training Results', fontsize=16)
        
        # Model accuracy comparison
        model_names = list(self.model_scores.keys())
        accuracies = [self.model_scores[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Individual Model Accuracy')
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training time comparison
        training_times = [self.model_scores[name]['training_time'] for name in model_names]
        
        axes[0, 1].bar(model_names, training_times, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Training Time per Model')
        axes[0, 1].set_xlabel('Model')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model parameters comparison
        parameters = [self.model_scores[name]['parameters'] for name in model_names]
        
        axes[1, 0].bar(model_names, parameters, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Model Parameters')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Parameters')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Metrics comparison
        metrics = ['accuracy', 'precision', 'recall']
        x_pos = np.arange(len(metrics))
        
        ensemble_metrics = [
            self.ensemble_results['accuracy'],
            self.ensemble_results['precision'],
            self.ensemble_results['recall']
        ]
        
        axes[1, 1].bar(x_pos, ensemble_metrics, color='gold', alpha=0.7)
        axes[1, 1].set_title('Ensemble Performance')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(metrics, rotation=45)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_training_results.png', dpi=300, bbox_inches='tight')
        print("📊 Ensemble results plot saved as 'ensemble_training_results.png'")
        plt.show()

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 Multi-Model Ensemble Gender Detection Training")
    print("Train multiple architectures for maximum performance")
    print("=" * 70)
    
    # Create trainer
    trainer = EnsembleModelTrainer(
        data_dir='gender_dataset_face',
        img_size=224,
        batch_size=32,
        epochs=40
    )
    
    try:
        # Train all models
        models, scores = trainer.train_all_models()
        
        print(f"\n🎉 Ensemble training completed successfully!")
        print(f"📁 Files saved:")
        print(f"   - gender_detection_*.keras (individual models)")
        print(f"   - ensemble_training_results.json (results)")
        print(f"   - ensemble_training_results.png (plots)")
        print(f"   - best_*_model.keras (best checkpoints)")
        
        if len(models) > 1:
            print(f"\n🏆 Best performing model: {max(scores.items(), key=lambda x: x[1]['accuracy'])[0]}")
            print(f"🎯 Ensemble accuracy: {trainer.ensemble_results['accuracy']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

