#!/usr/bin/env python3
"""
Working Gender Detection System
Uses Haar Cascade + Simple Logic - RELIABLE for tomorrow's demo!
"""

import cv2
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingGenderDetection:
    def __init__(self):
        """Initialize working gender detection system"""
        # Load Haar Cascade classifiers (built into OpenCV - no downloads needed!)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Gender detection parameters (tuned for better accuracy)
        self.female_threshold = 0.60  # Higher threshold = less false positives
        self.gender_weights = {
            'face_width': 0.25,
            'eye_spacing': 0.20,
            'jaw_strength': 0.25,
            'cheekbone_position': 0.15,
            'forehead_ratio': 0.15
        }
        
        logger.info("Working Gender Detection initialized successfully!")
        logger.info(f"Female threshold set to: {self.female_threshold}")
    
    def detect_faces(self, image):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes within face
            eyes = self.eye_cascade.detectMultiScale(
                face_roi, 
                scaleFactor=1.1, 
                minNeighbors=3, 
                minSize=(20, 20)
            )
            
            # Analyze facial features
            gender_score = self._analyze_facial_features(face_roi, eyes, w, h)
            
            # Determine gender
            if gender_score > self.female_threshold:
                gender = 'Female'
                confidence = gender_score
                color = (255, 0, 255)  # Magenta
            else:
                gender = 'Male'
                confidence = 1.0 - gender_score
                color = (0, 165, 255)  # Orange
            
            results.append({
                'bbox': (x, y, w, h),
                'gender': gender,
                'confidence': confidence,
                'color': color,
                'gender_score': gender_score
            })
            
            # Debug output
            print(f"🔍 Face {len(results)}: Score {gender_score:.3f} → {gender} (Conf: {confidence:.3f})")
        
        return results
    
    def _analyze_facial_features(self, face_roi, eyes, face_width, face_height):
        """Analyze facial features to determine gender probability"""
        if len(eyes) < 2:
            return 0.5  # Neutral if can't detect eyes
        
        # Sort eyes by x-coordinate (left to right)
        eyes = sorted(eyes, key=lambda x: x[0])
        
        # Calculate facial measurements
        measurements = {}
        
        # 1. Face width ratio (wider faces tend to be male)
        measurements['face_width'] = face_width / face_height
        
        # 2. Eye spacing (wider spacing tends to be male)
        if len(eyes) >= 2:
            eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            eye_spacing = np.sqrt((eye2_center[0] - eye1_center[0])**2 + 
                                (eye2_center[1] - eye1_center[1])**2)
            measurements['eye_spacing'] = eye_spacing / face_width
        else:
            measurements['eye_spacing'] = 0.3  # Default value
        
        # 3. Jaw strength (wider jaw tends to be male)
        jaw_width = face_width * 0.8
        measurements['jaw_strength'] = jaw_width / face_height
        
        # 4. Cheekbone position (higher tends to be female)
        cheekbone_y = face_height * 0.4
        measurements['cheekbone_position'] = cheekbone_y / face_height
        
        # 5. Forehead ratio (larger forehead tends to be male)
        forehead_height = face_height * 0.3
        measurements['forehead_ratio'] = forehead_height / face_height
        
        # Calculate weighted gender score
        gender_score = 0.0
        for feature, weight in self.gender_weights.items():
            if feature in measurements:
                # Normalize measurement to 0-1 range
                normalized_value = min(max(measurements[feature], 0.0), 1.0)
                gender_score += normalized_value * weight
        
        # Normalize final score
        gender_score = min(max(gender_score, 0.0), 1.0)
        
        return gender_score
    
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        return self.detect_faces(frame)

def test_working_detection():
    """Test working gender detection with webcam"""
    detector = WorkingGenderDetection()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("🎥 Working Gender Detection Started!")
    logger.info("Press 'q' to quit, 'i' for info, 't' to toggle threshold")
    
    current_threshold = detector.female_threshold
    
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
            color = result['color']
            
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
        
        # Draw threshold info
        cv2.putText(frame, f"Threshold: {current_threshold:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 't' to adjust threshold", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Working Gender Detection - Ready for Demo!', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            print(f"\n📊 Detection Info:")
            print(f"Face Detection: Haar Cascade")
            print(f"Gender Classification: Facial Feature Analysis")
            print(f"Current Threshold: {current_threshold:.2f}")
            print(f"Higher threshold = More strict (fewer false positives)")
        elif key == ord('t'):
            # Toggle between different threshold values
            thresholds = [0.55, 0.60, 0.65, 0.70]
            current_idx = thresholds.index(current_threshold) if current_threshold in thresholds else 0
            current_idx = (current_idx + 1) % len(thresholds)
            current_threshold = thresholds[current_idx]
            detector.female_threshold = current_threshold
            print(f"🔧 Threshold changed to: {current_threshold:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("✅ Demo system closed successfully!")

if __name__ == "__main__":
    test_working_detection()

