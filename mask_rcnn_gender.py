#!/usr/bin/env python3
"""
Mask R-CNN Gender Detection Implementation
Advanced face detection and gender classification with instance segmentation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class MaskRCNNGenderDetector:
    """
    Mask R-CNN implementation for gender detection with face segmentation
    """
    
    def __init__(self, 
                 img_size=512,
                 num_classes=2,
                 backbone='resnet50'):
        
        self.img_size = img_size
        self.num_classes = num_classes + 1  # +1 for background
        self.backbone = backbone
        self.class_names = ['background', 'man', 'woman']
        
        # Anchor configuration
        self.anchor_scales = [32, 64, 128, 256, 512]
        self.anchor_ratios = [0.5, 1.0, 2.0]
        
        print(f"🎯 Mask R-CNN Gender Detector")
        print(f"📐 Image size: {img_size}x{img_size}")
        print(f"🏗️  Backbone: {backbone}")
        print(f"👥 Classes: {self.class_names}")
        
    def create_backbone(self):
        """Create ResNet50 backbone for feature extraction"""
        if self.backbone == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3)
            )
            
            # Extract feature maps from different stages
            c2_output = backbone.get_layer('conv2_block3_out').output  # 128x128
            c3_output = backbone.get_layer('conv3_block4_out').output  # 64x64
            c4_output = backbone.get_layer('conv4_block6_out').output  # 32x32
            c5_output = backbone.get_layer('conv5_block3_out').output  # 16x16
            
            return Model(
                inputs=backbone.input,
                outputs=[c2_output, c3_output, c4_output, c5_output]
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
    
    def create_fpn(self, backbone_features):
        """Create Feature Pyramid Network (FPN)"""
        c2, c3, c4, c5 = backbone_features
        
        # Top-down pathway
        p5 = layers.Conv2D(256, 1, name='fpn_c5p5')(c5)
        p4 = layers.Add(name='fpn_p4add')([
            layers.UpSampling2D(2, name='fpn_p5upsampled')(p5),
            layers.Conv2D(256, 1, name='fpn_c4p4')(c4)
        ])
        p3 = layers.Add(name='fpn_p3add')([
            layers.UpSampling2D(2, name='fpn_p4upsampled')(p4),
            layers.Conv2D(256, 1, name='fpn_c3p3')(c3)
        ])
        p2 = layers.Add(name='fpn_p2add')([
            layers.UpSampling2D(2, name='fpn_p3upsampled')(p3),
            layers.Conv2D(256, 1, name='fpn_c2p2')(c2)
        ])
        
        # Apply 3x3 convolution to reduce aliasing
        p2 = layers.Conv2D(256, 3, padding='same', name='fpn_p2')(p2)
        p3 = layers.Conv2D(256, 3, padding='same', name='fpn_p3')(p3)
        p4 = layers.Conv2D(256, 3, padding='same', name='fpn_p4')(p4)
        p5 = layers.Conv2D(256, 3, padding='same', name='fpn_p5')(p5)
        
        # Additional level for P6
        p6 = layers.MaxPooling2D(pool_size=1, strides=2, name='fpn_p6')(p5)
        
        return [p2, p3, p4, p5, p6]
    
    def rpn_class_loss_graph(self, rpn_match, rpn_class_logits):
        """RPN classification loss"""
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(rpn_match, -1)
        
        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.not_equal(rpn_match, 0))
        
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        
        # Cross entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            anchor_class, rpn_class_logits, from_logits=True
        )
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        return loss
    
    def rpn_bbox_loss_graph(self, target_bbox, rpn_match, rpn_bbox):
        """RPN bounding box regression loss"""
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(rpn_match, -1)
        
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        indices = tf.where(tf.equal(rpn_match, 1))
        
        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)
        
        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = batch_pack_graph(target_bbox, batch_counts, tf.size(rpn_bbox) // 4)
        
        loss = smooth_l1_loss(target_bbox, rpn_bbox)
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        return loss
    
    def create_rpn(self, feature_maps):
        """Create Region Proposal Network (RPN)"""
        rpn_outputs = []
        
        for i, feature_map in enumerate(feature_maps):
            # Shared convolutional layer
            shared = layers.Conv2D(
                512, 3, padding='same', activation='relu',
                name=f'rpn_conv_shared_{i}'
            )(feature_map)
            
            # Classification branch (object vs background)
            rpn_class_raw = layers.Conv2D(
                len(self.anchor_ratios) * 2, 1,
                name=f'rpn_class_raw_{i}'
            )(shared)
            
            rpn_class_logits = layers.Reshape(
                (-1, 2), name=f'rpn_class_logits_{i}'
            )(rpn_class_raw)
            
            rpn_probs = layers.Activation(
                'softmax', name=f'rpn_class_{i}'
            )(rpn_class_logits)
            
            # Bounding box regression branch
            rpn_bbox = layers.Conv2D(
                len(self.anchor_ratios) * 4, 1,
                name=f'rpn_bbox_{i}'
            )(shared)
            
            rpn_bbox = layers.Reshape(
                (-1, 4), name=f'rpn_bbox_reshape_{i}'
            )(rpn_bbox)
            
            rpn_outputs.append([rpn_class_logits, rpn_probs, rpn_bbox])
        
        return rpn_outputs
    
    def create_classifier_head(self, rois, feature_maps, pool_size=7):
        """Create classifier and mask heads"""
        # ROI Pooling/Align
        pooled_features = self.roi_align(rois, feature_maps, pool_size)
        
        # Shared layers
        x = layers.TimeDistributed(
            layers.Conv2D(1024, pool_size, padding='valid'),
            name='mrcnn_class_conv1'
        )(pooled_features)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='mrcnn_class_bn1'
        )(x)
        x = layers.Activation('relu')(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(1024, 1),
            name='mrcnn_class_conv2'
        )(x)
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='mrcnn_class_bn2'
        )(x)
        x = layers.Activation('relu')(x)
        
        x = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(),
            name='mrcnn_class_pool'
        )(x)
        
        # Classification head
        mrcnn_class_logits = layers.TimeDistributed(
            layers.Dense(self.num_classes),
            name='mrcnn_class_logits'
        )(x)
        
        mrcnn_probs = layers.TimeDistributed(
            layers.Activation('softmax'),
            name='mrcnn_class'
        )(mrcnn_class_logits)
        
        # Bounding box head
        mrcnn_bbox = layers.TimeDistributed(
            layers.Dense(self.num_classes * 4, activation='linear'),
            name='mrcnn_bbox'
        )(x)
        
        # Mask head
        mask_features = layers.TimeDistributed(
            layers.Conv2DTranspose(256, 2, strides=2, activation='relu'),
            name='mrcnn_mask_deconv'
        )(pooled_features)
        
        mrcnn_mask = layers.TimeDistributed(
            layers.Conv2D(self.num_classes, 1, activation='sigmoid'),
            name='mrcnn_mask'
        )(mask_features)
        
        return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_mask
    
    def roi_align(self, rois, feature_maps, pool_size):
        """ROI Align implementation"""
        # Simplified ROI Align - in practice, you'd use tf.image.crop_and_resize
        # This is a placeholder implementation
        batch_size = tf.shape(rois)[0]
        num_rois = tf.shape(rois)[1]
        
        # Use the first feature map for simplicity
        feature_map = feature_maps[0]
        
        # Create pooled features tensor
        pooled = tf.zeros((batch_size, num_rois, pool_size, pool_size, 256))
        
        return pooled
    
    def build_model(self):
        """Build the complete Mask R-CNN model"""
        # Input
        input_image = layers.Input(
            shape=(self.img_size, self.img_size, 3),
            name='input_image'
        )
        
        # Backbone
        backbone = self.create_backbone()
        backbone_features = backbone(input_image)
        
        # Feature Pyramid Network
        fpn_features = self.create_fpn(backbone_features)
        
        # Region Proposal Network
        rpn_outputs = self.create_rpn(fpn_features)
        
        # Combine RPN outputs
        rpn_class_logits = layers.Concatenate(
            axis=1, name='rpn_class_logits'
        )([output[0] for output in rpn_outputs])
        
        rpn_probs = layers.Concatenate(
            axis=1, name='rpn_class'
        )([output[1] for output in rpn_outputs])
        
        rpn_bbox = layers.Concatenate(
            axis=1, name='rpn_bbox'
        )([output[2] for output in rpn_outputs])
        
        # For training, we need additional inputs for ground truth
        if self.training:
            # Training inputs
            input_gt_class_ids = layers.Input(
                shape=[None], name='input_gt_class_ids', dtype=tf.int32
            )
            input_gt_boxes = layers.Input(
                shape=[None, 4], name='input_gt_boxes', dtype=tf.float32
            )
            input_gt_masks = layers.Input(
                shape=[None, self.img_size, self.img_size],
                name='input_gt_masks', dtype=tf.bool
            )
            
            # Proposal layer (generates ROIs from RPN outputs)
            active_class_ids = layers.Lambda(
                lambda x: self.parse_image_meta_graph(x)['active_class_ids']
            )(input_image_meta)
            
            # Create model
            model = Model(
                inputs=[input_image, input_image_meta, input_gt_class_ids, 
                       input_gt_boxes, input_gt_masks],
                outputs=[rpn_class_logits, rpn_probs, rpn_bbox],
                name='mask_rcnn_training'
            )
        else:
            # Inference model
            model = Model(
                inputs=input_image,
                outputs=[rpn_class_logits, rpn_probs, rpn_bbox],
                name='mask_rcnn_inference'
            )
        
        return model
    
    def compile_model(self, model):
        """Compile the model with appropriate losses"""
        # Custom loss functions
        def rpn_class_loss(y_true, y_pred):
            return self.rpn_class_loss_graph(y_true, y_pred)
        
        def rpn_bbox_loss(y_true, y_pred):
            return self.rpn_bbox_loss_graph(y_true, y_pred)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'rpn_class_logits': rpn_class_loss,
                'rpn_bbox': rpn_bbox_loss,
            },
            metrics=['accuracy']
        )
        
        return model

class SimplifiedGenderMaskRCNN:
    """
    Simplified Mask R-CNN implementation for gender detection
    Focuses on the core concepts without full complexity
    """
    
    def __init__(self, img_size=224, num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_names = ['man', 'woman']
        
    def create_model(self):
        """Create a simplified model inspired by Mask R-CNN concepts"""
        # Input
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Backbone (ResNet50-like)
        backbone = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        
        # Feature extraction
        x = backbone.output
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers for classification
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train the simplified model"""
        model = self.create_model()
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint('simplified_mask_rcnn_gender.keras', save_best_only=True)
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history

def smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss for bounding box regression"""
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row"""
    outputs = []
    for i in range(tf.shape(counts)[0]):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def main():
    """Main function to demonstrate Mask R-CNN concepts"""
    print("🎯 Mask R-CNN Gender Detection Demo")
    print("=" * 50)
    
    # For this demo, we'll use the simplified version
    print("Creating simplified Mask R-CNN inspired model...")
    
    model_builder = SimplifiedGenderMaskRCNN(img_size=224, num_classes=2)
    model = model_builder.create_model()
    
    print("✅ Model created successfully!")
    print(f"📊 Model summary:")
    model.summary()
    
    # Save model architecture
    model.save('mask_rcnn_gender_architecture.keras')
    print("💾 Model architecture saved as 'mask_rcnn_gender_architecture.keras'")
    
    print("\n🔧 To use this model:")
    print("1. Load your dataset using the training script")
    print("2. Train the model with your data")
    print("3. Use the inference script for detection")

if __name__ == "__main__":
    main()
