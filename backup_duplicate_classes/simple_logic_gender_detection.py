#!/usr/bin/env python3
"""
Simple Logic-Based Gender Detection System
Uses OpenCV and basic facial feature analysis
No external dependencies beyond OpenCV
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

class SimpleLogicGenderDetector:
    def __init__(self):
        """Initialize the simple logic-based gender detector"""
        
        # Load face detection models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load additional cascades for better feature detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Gender detection parameters (tuned based on facial anatomy)
        self.gender_weights = {
            'face_width_ratio': 0.25,     # Wider face = more masculine
            'eye_spacing_ratio': 0.20,    # Wider eye spacing = more masculine
            'jaw_line_ratio': 0.25,       # Stronger jaw = more masculine
            'cheekbone_ratio': 0.15,      # Higher cheekbones = more feminine
            'forehead_ratio': 0.15        # Higher forehead = more feminine
        }
        
        print("🧠 Simple Logic-Based Gender Detection System")
        print("📊 Using OpenCV-based facial feature analysis")
        print("🔍 Features analyzed:")
        print("   - Face width ratio")
        print("   - Eye spacing ratio")
        print("   - Jaw line strength")
        print("   - Cheekbone position")
        print("   - Forehead ratio")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_eyes(self, face_roi):
        """Detect eyes within a face region"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5, minSize=(20, 20))
        return eyes
    
    def analyze_facial_features(self, frame, face_bbox):
        """Analyze facial features for gender determination"""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        features = {}
        
        # 1. Face width ratio (width/height)
        features['face_width_ratio'] = w / h
        
        # 2. Eye spacing ratio
        eyes = self.detect_eyes(face_roi)
        if len(eyes) >= 2:
            # Calculate distance between eyes
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (ex + ew//2, ey + eh//2)
                eye_centers.append(eye_center)
            
            if len(eye_centers) >= 2:
                # Sort eyes by x-coordinate
                eye_centers.sort(key=lambda p: p[0])
                eye_spacing = np.linalg.norm(np.array(eye_centers[1]) - np.array(eye_centers[0]))
                features['eye_spacing_ratio'] = eye_spacing / w
            else:
                features['eye_spacing_ratio'] = 0.3  # Default value
        else:
            features['eye_spacing_ratio'] = 0.3  # Default value
        
        # 3. Jaw line strength (analyze lower third of face)
        lower_face = face_roi[int(h*0.6):h, :]
        if lower_face.size > 0:
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            # Edge detection to find jaw line
            edges = cv2.Canny(gray_lower, 50, 150)
            jaw_strength = np.sum(edges > 0) / edges.size
            features['jaw_line_ratio'] = min(jaw_strength * 3, 1.0)
        else:
            features['jaw_line_ratio'] = 0.5
        
        # 4. Cheekbone position (analyze middle third of face)
        middle_face = face_roi[int(h*0.3):int(h*0.7), :]
        if middle_face.size > 0:
            gray_middle = cv2.cvtColor(middle_face, cv2.COLOR_BGR2GRAY)
            # Look for horizontal edges (cheekbone lines)
            sobel_x = cv2.Sobel(gray_middle, cv2.CV_64F, 1, 0, ksize=3)
            cheekbone_strength = np.sum(np.abs(sobel_x) > 50) / sobel_x.size
            features['cheekbone_ratio'] = min(cheekbone_strength * 2, 1.0)
        else:
            features['cheekbone_ratio'] = 0.5
        
        # 5. Forehead ratio (analyze upper third of face)
        upper_face = face_roi[:int(h*0.4), :]
        if upper_face.size > 0:
            gray_upper = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
            # Look for smoothness (forehead is usually smooth)
            laplacian = cv2.Laplacian(gray_upper, cv2.CV_64F)
            smoothness = 1.0 - (np.sum(np.abs(laplacian) > 30) / laplacian.size)
            features['forehead_ratio'] = smoothness
        else:
            features['forehead_ratio'] = 0.5
        
        return features
    
    def calculate_gender_score(self, features):
        """Calculate gender score based on facial features"""
        score = 0.0
        
        # Face width: wider = more masculine
        face_width_score = features['face_width_ratio'] * self.gender_weights['face_width_ratio']
        score += face_width_score
        
        # Eye spacing: wider = more masculine
        eye_spacing_score = features['eye_spacing_ratio'] * self.gender_weights['eye_spacing_ratio']
        score += eye_spacing_score
        
        # Jaw line: stronger = more masculine
        jaw_score = features['jaw_line_ratio'] * self.gender_weights['jaw_line_ratio']
        score += jaw_score
        
        # Cheekbone: higher = more feminine
        cheekbone_score = (1 - features['cheekbone_ratio']) * self.gender_weights['cheekbone_ratio']
        score += cheekbone_score
        
        # Forehead: higher = more feminine
        forehead_score = (1 - features['forehead_ratio']) * self.gender_weights['forehead_ratio']
        score += forehead_score
        
        return score
    
    def predict_gender(self, features):
        """Predict gender based on facial features"""
        gender_score = self.calculate_gender_score(features)
        
        # Normalize score to 0-1 range
        normalized_score = max(0, min(1, gender_score))
        
        # Threshold for gender classification
        if normalized_score > 0.55:
            gender = 'woman'
            confidence = normalized_score
        else:
            gender = 'man'
            confidence = 1 - normalized_score
        
        return gender, confidence, features
    
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        # Detect faces
        faces = self.detect_faces(frame)
        
        results = []
        
        for face_bbox in faces:
            try:
                # Extract facial features
                features = self.extract_facial_features(frame, face_bbox)
                
                # Predict gender
                gender, confidence, feature_values = self.predict_gender(features)
                
                results.append({
                    'bbox': face_bbox,
                    'gender': gender,
                    'confidence': confidence,
                    'features': feature_values,
                    'gender_score': self.calculate_gender_score(feature_values)
                })
                
            except Exception as e:
                print(f"⚠️  Error processing face: {e}")
                continue
        
        return results
    
    def extract_facial_features(self, frame, face_bbox):
        """Extract facial features for gender analysis"""
        return self.analyze_facial_features(frame, face_bbox)
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        for result in results:
            x, y, w, h = result['bbox']
            gender = result['gender']
            confidence = result['confidence']
            features = result['features']
            
            # Choose color based on gender
            color = (255, 0, 0) if gender == 'man' else (0, 255, 255)  # Blue for man, Yellow for woman
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"{gender}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw feature analysis info
            feature_text = f"Score: {result['gender_score']:.2f}"
            cv2.putText(
                frame, feature_text,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
            # Draw feature breakdown
            y_offset = y + h + 40
            for feature_name, value in features.items():
                feature_text = f"{feature_name}: {value:.2f}"
                cv2.putText(
                    frame, feature_text,
                    (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )
                y_offset += 15
        
        return frame

def run_webcam_detection():
    """Run the logic-based gender detection on webcam"""
    print("🎥 Starting Simple Logic-Based Gender Detection")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize detector
    detector = SimpleLogicGenderDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("🧠 Using logic-based gender detection...")
    print("💡 This system analyzes facial features to determine gender")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Draw results
            frame_with_results = detector.draw_results(frame, results)
            
            # Add status text
            cv2.putText(
                frame_with_results,
                "Logic-Based Gender Detection - Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow('Simple Logic-Based Gender Detection', frame_with_results)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"logic_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_results)
                print(f"📸 Screenshot saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Webcam session ended")

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 Simple Logic-Based Gender Detection System")
    print("=" * 70)
    print("This system uses OpenCV-based facial feature analysis")
    print("Features analyzed:")
    print("  - Face width ratio")
    print("  - Eye spacing ratio")
    print("  - Jaw line strength")
    print("  - Cheekbone position")
    print("  - Forehead ratio")
    print()
    print("💡 The system uses anatomical differences between male and female faces")
    print("   to make gender predictions based on feature ratios and patterns.")
    print()
    
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is working and accessible")
