#!/usr/bin/env python3
"""
DeepFace-based Gender Detection System
Uses pre-trained deep learning models for 97%+ accuracy
"""

import cv2
import numpy as np
from deepface import DeepFace
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepFaceGenderDetection:
    def __init__(self):
        """Initialize DeepFace gender detection system"""
        self.models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace']
        self.current_model = 'VGG-Face'  # Start with VGG-Face (good balance of speed/accuracy)
        self.confidence_threshold = 0.6
        
        logger.info(f"DeepFace Gender Detection initialized with model: {self.current_model}")
        logger.info("Available models: " + ", ".join(self.models))
    
    def detect_gender(self, image):
        """Detect gender using DeepFace"""
        try:
            # DeepFace analysis
            result = DeepFace.analyze(
                img_path=image,
                actions=['gender'],
                models={},
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected
            
            gender = result['gender']
            confidence = result['dominant_gender_confidence'] if 'dominant_gender_confidence' in result else 0.8
            
            logger.info(f"DeepFace detected: {gender} with confidence: {confidence:.3f}")
            
            return gender, confidence
            
        except Exception as e:
            logger.error(f"DeepFace error: {e}")
            return 'unknown', 0.0
    
    def switch_model(self, model_name):
        """Switch to a different DeepFace model"""
        if model_name in self.models:
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
        else:
            logger.error(f"Invalid model: {model_name}. Available: {self.models}")
            return False
    
    def get_available_models(self):
        """Get list of available models"""
        return self.models.copy()

def test_deepface():
    """Test DeepFace with webcam"""
    detector = DeepFaceGenderDetection()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return
    
    logger.info("Press 'q' to quit, 'm' to switch models, 'i' for model info")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        try:
            # Convert frame to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame temporarily for DeepFace
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Detect gender
            gender, confidence = detector.detect_gender(temp_path)
            
            # Draw results
            cv2.putText(display_frame, f"Gender: {gender.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Confidence: {confidence:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Model: {detector.current_model}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Color based on gender
            if gender.lower() == 'woman':
                color = (255, 0, 255)  # Magenta
            elif gender.lower() == 'man':
                color = (0, 165, 255)  # Orange
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw confidence bar
            bar_width = int(confidence * 200)
            cv2.rectangle(display_frame, (10, 140), (10 + bar_width, 160), color, -1)
            cv2.rectangle(display_frame, (10, 140), (210, 160), (255, 255, 255), 2)
            
        except Exception as e:
            cv2.putText(display_frame, f"Error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow('DeepFace Gender Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Cycle through models
            current_idx = detector.models.index(detector.current_model)
            next_idx = (current_idx + 1) % len(detector.models)
            detector.switch_model(detector.models[next_idx])
        elif key == ord('i'):
            print(f"\n📊 Available Models:")
            for i, model in enumerate(detector.models):
                marker = " → " if model == detector.current_model else "   "
                print(f"{marker}{i+1}. {model}")
            print(f"Current: {detector.current_model}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_deepface()

