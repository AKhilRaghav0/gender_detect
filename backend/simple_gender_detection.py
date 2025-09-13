#!/usr/bin/env python3
"""
Simple Gender Detection using OpenCV DNN
Uses pre-trained Caffe models for gender classification
No TensorFlow dependencies - should work reliably
"""

import cv2
import numpy as np
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleGenderDetection:
    def __init__(self):
        """Initialize simple gender detection system"""
        self.face_net = None
        self.gender_net = None
        self.gender_list = ['Male', 'Female']
        
        # Model paths (will download if not present)
        self.face_model_path = "opencv_face_detector_uint8.pb"
        self.face_config_path = "opencv_face_detector.pbtxt"
        self.gender_model_path = "gender_net.caffemodel"
        self.gender_config_path = "gender_deploy.prototxt"
        
        # Model URLs for download
        self.model_urls = {
            'face_model': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'face_config': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'gender_model': 'https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ',
            'gender_config': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/gender_deploy.prototxt'
        }
        
        self.face_size = (300, 300)
        self.gender_size = (227, 227)
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize face detection and gender classification models"""
        try:
            # Face detection model
            if not os.path.exists(self.face_model_path):
                logger.info("Downloading face detection model...")
                self._download_model('face_model', self.face_model_path)
            
            if not os.path.exists(self.face_config_path):
                logger.info("Downloading face detection config...")
                self._download_model('face_config', self.face_config_path)
            
            # Gender classification model
            if not os.path.exists(self.gender_model_path):
                logger.info("Downloading gender classification model...")
                self._download_model('gender_model', self.gender_model_path)
            
            if not os.path.exists(self.gender_config_path):
                logger.info("Downloading gender classification config...")
                self._download_model('gender_config', self.gender_config_path)
            
            # Load models
            self.face_net = cv2.dnn.readNet(self.face_model_path, self.face_config_path)
            self.gender_net = cv2.dnn.readNet(self.gender_model_path, self.gender_config_path)
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.info("Falling back to Haar Cascade method...")
            self._fallback_initialization()
    
    def _download_model(self, model_key, file_path):
        """Download model files (simplified version)"""
        try:
            import urllib.request
            urllib.request.urlretrieve(self.model_urls[model_key], file_path)
            logger.info(f"Downloaded {file_path}")
        except Exception as e:
            logger.warning(f"Could not download {file_path}: {e}")
            logger.info("Please download manually or use fallback method")
    
    def _fallback_initialization(self):
        """Fallback to Haar Cascade method if DNN fails"""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        logger.info("Fallback to Haar Cascade initialized")
    
    def detect_faces_dnn(self, image):
        """Detect faces using DNN model"""
        if self.face_net is None:
            return self.detect_faces_haar(image)
        
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, self.face_size, (104, 177, 123), False, False)
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                faces.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': confidence
                })
        
        return faces
    
    def detect_faces_haar(self, image):
        """Detect faces using Haar Cascade (fallback)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return [{'bbox': (x, y, w, h), 'confidence': 0.8} for (x, y, w, h) in faces]
    
    def classify_gender(self, face_roi):
        """Classify gender using DNN model"""
        if self.gender_net is None:
            return self._classify_gender_simple(face_roi)
        
        try:
            # Preprocess face ROI
            face_blob = cv2.dnn.blobFromImage(face_roi, 1.0, self.gender_size, self.mean_values, False, False)
            
            # Gender prediction
            self.gender_net.setInput(face_blob)
            gender_preds = self.gender_net.forward()
            
            gender = self.gender_list[gender_preds[0].argmax()]
            confidence = float(gender_preds[0].max())
            
            return gender, confidence
            
        except Exception as e:
            logger.warning(f"Gender classification failed: {e}")
            return self._classify_gender_simple(face_roi)
    
    def _classify_gender_simple(self, face_roi):
        """Simple gender classification using facial features"""
        height, width = face_roi.shape[:2]
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristics (this is where you'd use a real ML model)
        # For now, using basic ratios as placeholder
        
        # Face aspect ratio
        aspect_ratio = width / height
        
        # Simple classification (this is just a demo - replace with real model)
        if aspect_ratio > 0.8:  # Wider face
            gender = 'Male'
            confidence = 0.7
        else:  # Narrower face
            gender = 'Female'
            confidence = 0.7
        
        return gender, confidence
    
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        # Detect faces
        faces = self.detect_faces_dnn(frame)
        
        results = []
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Classify gender
                gender, confidence = self.classify_gender(face_roi)
                
                results.append({
                    'bbox': face['bbox'],
                    'gender': gender,
                    'confidence': confidence,
                    'face_confidence': face['confidence']
                })
        
        return results

def test_simple_detection():
    """Test simple gender detection with webcam"""
    detector = SimpleGenderDetection()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("Press 'q' to quit, 'i' for info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        results = detector.process_frame(frame)
        
        # Draw results
        for result in results:
            x, y, w, h = result['bbox']
            gender = result['gender']
            confidence = result['confidence']
            
            # Color based on gender
            if gender == 'Female':
                color = (255, 0, 255)  # Magenta
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw gender label
            label = f"{gender}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw confidence bar
            bar_width = int(confidence * 100)
            cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 20), color, -1)
            cv2.rectangle(frame, (x, y + h + 5), (x + 100, y + h + 20), (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Simple Gender Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            print(f"\n📊 Detection Info:")
            print(f"Face Detection: {'DNN' if detector.face_net else 'Haar Cascade'}")
            print(f"Gender Classification: {'DNN' if detector.gender_net else 'Simple Heuristics'}")
            print(f"Models Loaded: {len([x for x in [detector.face_net, detector.gender_net] if x is not None])}/2")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_simple_detection()

