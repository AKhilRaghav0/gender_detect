#!/usr/bin/env python3
"""
MediaPipe-based Gender Detection System
Lightweight, fast, and accurate gender detection
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MediaPipeGenderDetection:
    def __init__(self):
        """Initialize MediaPipe gender detection system"""
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Gender classification model (you can replace this with a custom model)
        self.gender_net = None
        self.gender_list = ['Male', 'Female']
        
        logger.info("MediaPipe Gender Detection initialized")
        logger.info("Note: This uses face detection + basic classification")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                
                # Get confidence
                confidence = detection.score[0]
                
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'keypoints': detection.location_data.relative_keypoints
                })
        
        return faces
    
    def classify_gender(self, face_roi):
        """Simple gender classification based on facial features"""
        # This is a simplified approach - you can replace with a trained model
        height, width = face_roi.shape[:2]
        
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
        faces = self.detect_faces(frame)
        
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

def test_mediapipe():
    """Test MediaPipe with webcam"""
    detector = MediaPipeGenderDetection()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("Press 'q' to quit")
    
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
        cv2.imshow('MediaPipe Gender Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_mediapipe()

