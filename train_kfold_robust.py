#!/usr/bin/env python3
"""
K-Fold Cross-Validation Training Script
Maximum robustness for production deployment
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub

class RobustGenderDetector:
    def __init__(self, 
                 data_dir='gender_dataset_face',
                 img_size=224,
                 batch_size=32,
                 epochs=30,
                 k_folds=5,
                 model_type='resnet50'):
        
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_folds = k_folds
        self.model_type = model_type
        self.models = []
        self.histories = []
        self.fold_scores = []
        
        # Class mapping
        self.class_names = ['man', 'woman']
        self.num_classes = len(self.class_names)
        
        print(f"🚀 Robust K-Fold Gender Detection Training")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"🖼️  Image size: {self.img_size}x{self.img_size}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔄 Epochs per fold: {self.epochs}")
        print(f"🎯 K-Folds: {self.k_folds}")
        print(f"🏗️  Model type: {self.model_type}")
        
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
                            print("  🚀 GH200 detected - Enabling mixed precision + XLA")
                            tf.keras.mixed_precision.set_global_policy('mixed_float16')
                            # Enable XLA for maximum speed
                            tf.config.optimizer.set_jit(True)
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
        """Load all data for K-fold splitting"""
        print("📊 Loading dataset for K-fold splitting...")
        
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
        
        return X, y
    
    def create_model(self, fold_num):
        """Create model based on type"""
        print(f"🏗️  Creating {self.model_type} model for fold {fold_num + 1}...")
        
        if self.model_type == 'resnet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        elif self.model_type == 'densenet':
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        else:
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
        
        # Freeze base model
        base_model.trainable = False
        
        # Create full model with advanced architecture
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
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
        
        # Compile with advanced optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        print(f"✅ Model created with {model.count_params():,} parameters")
        return model
    
    def create_callbacks(self, fold_num):
        """Create advanced callbacks for each fold"""
        return [
            # Early stopping with patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduling
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                f'best_model_fold_{fold_num + 1}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/fold_{fold_num + 1}',
                histogram_freq=1,
                write_graph=True
            )
        ]
    
    def train_fold(self, fold_num, X_train, X_val, y_train, y_val):
        """Train a single fold"""
        print(f"\n🎯 Training Fold {fold_num + 1}/{self.k_folds}")
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Validation samples: {len(X_val)}")
        
        # Create model for this fold
        model = self.create_model(fold_num)
        
        # Create callbacks
        callback_list = self.create_callbacks(fold_num)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1
        )
        
        # Load best model
        model = keras.models.load_model(f'best_model_fold_{fold_num + 1}.keras')
        
        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(
            X_val, y_val, verbose=0
        )
        
        # Store results
        fold_score = {
            'fold': fold_num + 1,
            'accuracy': val_acc,
            'precision': val_precision,
            'recall': val_recall,
            'auc': val_auc,
            'loss': val_loss
        }
        
        self.fold_scores.append(fold_score)
        self.models.append(model)
        self.histories.append(history)
        
        print(f"✅ Fold {fold_num + 1} completed:")
        print(f"   Accuracy: {val_acc:.4f}")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall: {val_recall:.4f}")
        print(f"   AUC: {val_auc:.4f}")
        
        return model, history
    
    def train_kfold(self):
        """Train using K-fold cross-validation"""
        # Setup GPU
        self.setup_gpu()
        
        # Load data
        X, y = self.load_data()
        
        # Initialize K-fold
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        print(f"\n🔄 Starting {self.k_folds}-fold cross-validation training...")
        
        # Train each fold
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Convert to categorical
            y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
            y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)
            
            # Train this fold
            self.train_fold(fold_num, X_train, X_val, y_train, y_val)
        
        # Calculate ensemble results
        self.calculate_ensemble_results()
        
        # Save ensemble model
        self.save_ensemble_model()
        
        # Plot results
        self.plot_kfold_results()
        
        return self.models, self.fold_scores
    
    def calculate_ensemble_results(self):
        """Calculate ensemble performance metrics"""
        print("\n📊 Calculating ensemble results...")
        
        # Load validation data for final evaluation
        X, y = self.load_data()
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        all_predictions = []
        all_true_labels = []
        
        for fold_num, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            # Get predictions from this fold's model
            predictions = self.models[fold_num].predict(X_val)
            all_predictions.extend(predictions)
            all_true_labels.extend(y_val)
        
        # Calculate ensemble metrics
        all_predictions = np.array(all_predictions)
        all_true_labels = np.array(all_true_labels)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(all_predictions, axis=0)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        ensemble_accuracy = accuracy_score(all_true_labels, ensemble_pred_classes)
        ensemble_precision = precision_score(all_true_labels, ensemble_pred_classes, average='weighted')
        ensemble_recall = recall_score(all_true_labels, ensemble_pred_classes, average='weighted')
        ensemble_f1 = f1_score(all_true_labels, ensemble_pred_classes, average='weighted')
        
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
            'f1_score': ensemble_f1
        }
    
    def save_ensemble_model(self):
        """Save ensemble model and results"""
        print("\n💾 Saving ensemble model and results...")
        
        # Save all individual models
        for i, model in enumerate(self.models):
            model.save(f'gender_detection_fold_{i+1}.keras')
        
        # Save ensemble results
        results = {
            'k_folds': self.k_folds,
            'model_type': self.model_type,
            'fold_scores': self.fold_scores,
            'ensemble_results': self.ensemble_results,
            'training_params': {
                'img_size': self.img_size,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        
        with open('kfold_ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Ensemble models and results saved!")
    
    def plot_kfold_results(self):
        """Plot K-fold training results"""
        print("\n📈 Generating K-fold results plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'K-Fold Cross-Validation Results ({self.k_folds} folds)', fontsize=16)
        
        # Fold accuracy comparison
        fold_nums = [score['fold'] for score in self.fold_scores]
        fold_accuracies = [score['accuracy'] for score in self.fold_scores]
        
        axes[0, 0].bar(fold_nums, fold_accuracies, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Accuracy per Fold')
        axes[0, 0].set_xlabel('Fold Number')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training history for first fold
        if self.histories:
            history = self.histories[0]
            axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Training History (Fold 1)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Metrics comparison across folds
        metrics = ['accuracy', 'precision', 'recall', 'auc']
        x_pos = np.arange(len(metrics))
        
        fold_metrics = []
        for metric in metrics:
            values = [score[metric] for score in self.fold_scores]
            fold_metrics.append(np.mean(values))
        
        axes[1, 0].bar(x_pos, fold_metrics, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average Metrics Across Folds')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(metrics, rotation=45)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss comparison
        fold_losses = [score['loss'] for score in self.fold_scores]
        axes[1, 1].bar(fold_nums, fold_losses, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('Loss per Fold')
        axes[1, 1].set_xlabel('Fold Number')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kfold_training_results.png', dpi=300, bbox_inches='tight')
        print("📊 K-fold results plot saved as 'kfold_training_results.png'")
        plt.show()

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 Robust K-Fold Gender Detection Training")
    print("Maximum robustness for production deployment")
    print("=" * 70)
    
    # Create trainer with different model types
    model_types = ['resnet50', 'efficientnet', 'densenet']
    
    for model_type in model_types:
        print(f"\n🎯 Training {model_type.upper()} model...")
        
        trainer = RobustGenderDetector(
            data_dir='gender_dataset_face',
            img_size=224,
            batch_size=32,
            epochs=30,
            k_folds=5,
            model_type=model_type
        )
        
        try:
            # Train K-fold
            models, scores = trainer.train_kfold()
            
            print(f"\n🎉 {model_type.upper()} training completed successfully!")
            print(f"📁 Files saved:")
            print(f"   - gender_detection_fold_1-5.keras (individual models)")
            print(f"   - kfold_ensemble_results.json (results)")
            print(f"   - kfold_training_results.png (plots)")
            
        except Exception as e:
            print(f"\n❌ {model_type.upper()} training failed: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
