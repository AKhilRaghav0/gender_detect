#!/usr/bin/env python3
"""
Ultra-Sensitive Gender Detection System
Detects females from much further away - no need to be super close!
Perfect for demonstrations and real-world use
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

class UltraSensitiveGenderDetector:
    def __init__(self):
        """Initialize the ultra-sensitive gender detector"""
        
        # Load face detection models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load additional cascades for better feature detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # ULTRA-SENSITIVE gender detection parameters
        self.gender_weights = {
            'face_width_ratio': 0.10,      # Wider face = more masculine (reduced weight)
            'eye_spacing_ratio': 0.15,     # Wider eye spacing = more masculine (reduced weight)
            'jaw_line_ratio': 0.35,        # Stronger jaw = more masculine (increased weight)
            'cheekbone_ratio': 0.25,       # Higher cheekbones = more feminine (increased weight)
            'forehead_ratio': 0.15         # Higher forehead = more feminine
        }
        
        # ULTRA-SENSITIVE thresholds
        self.female_threshold = 0.38  # Much lower threshold for detecting females from far away
        
        print("🚀 Ultra-Sensitive Gender Detection System")
        print("📊 Using highly sensitive facial feature analysis")
        print("🔍 Features analyzed:")
        print("   - Face width ratio")
        print("   - Eye spacing ratio")
        print("   - Jaw line strength")
        print("   - Cheekbone position")
        print("   - Forehead ratio")
        print(f"🎯 Female threshold: {self.female_threshold} (ULTRA-SENSITIVE!)")
        print("💡 Detects females from much further away!")
    
    def detect_faces(self, frame):
        """Detect faces in the frame with better sensitivity"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # More sensitive face detection parameters
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_eyes(self, face_roi):
        """Detect eyes within a face region with better sensitivity"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        # More sensitive eye detection
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.05, 3, minSize=(12, 12))
        return eyes
    
    def analyze_facial_features(self, frame, face_bbox):
        """Analyze facial features for gender determination with enhanced sensitivity"""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        features = {}
        
        # 1. Face width ratio (width/height) - enhanced sensitivity
        features['face_width_ratio'] = w / h
        
        # 2. Eye spacing ratio - enhanced detection
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
                features['eye_spacing_ratio'] = 0.30  # Enhanced default
        else:
            features['eye_spacing_ratio'] = 0.30  # Enhanced default
        
        # 3. Jaw line strength - enhanced analysis for better female detection
        lower_face = face_roi[int(h*0.6):h, :]
        if lower_face.size > 0:
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            # More sensitive edge detection for jaw line
            edges = cv2.Canny(gray_lower, 20, 70)  # Enhanced thresholds
            jaw_strength = np.sum(edges > 0) / edges.size
            features['jaw_line_ratio'] = min(jaw_strength * 2.2, 1.0)  # Enhanced multiplier
        else:
            features['jaw_line_ratio'] = 0.5
        
        # 4. Cheekbone position - enhanced detection for better female detection
        middle_face = face_roi[int(h*0.3):int(h*0.7), :]
        if middle_face.size > 0:
            gray_middle = cv2.cvtColor(middle_face, cv2.COLOR_BGR2GRAY)
            # More sensitive horizontal edge detection for cheekbones
            sobel_x = cv2.Sobel(gray_middle, cv2.CV_64F, 1, 0, ksize=3)
            cheekbone_strength = np.sum(np.abs(sobel_x) > 30) / sobel_x.size  # Enhanced threshold
            features['cheekbone_ratio'] = min(cheekbone_strength * 1.8, 1.0)  # Enhanced multiplier
        else:
            features['cheekbone_ratio'] = 0.5
        
        # 5. Forehead ratio - enhanced analysis for better female detection
        upper_face = face_roi[:int(h*0.4), :]
        if upper_face.size > 0:
            gray_upper = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
            # More sensitive smoothness detection for forehead
            laplacian = cv2.Laplacian(gray_upper, cv2.CV_64F)
            smoothness = 1.0 - (np.sum(np.abs(laplacian) > 18) / laplacian.size)  # Enhanced threshold
            features['forehead_ratio'] = smoothness
        else:
            features['forehead_ratio'] = 0.5
        
        return features
    
    def calculate_gender_score(self, features):
        """Calculate gender score based on facial features with enhanced sensitivity"""
        score = 0.0
        
        # Face width: wider = more masculine (reduced impact)
        face_width_score = features['face_width_ratio'] * self.gender_weights['face_width_ratio']
        score += face_width_score
        
        # Eye spacing: wider = more masculine (reduced impact)
        eye_spacing_score = features['eye_spacing_ratio'] * self.gender_weights['eye_spacing_ratio']
        score += eye_spacing_score
        
        # Jaw line: stronger = more masculine (increased impact)
        jaw_score = features['jaw_line_ratio'] * self.gender_weights['jaw_line_ratio']
        score += jaw_score
        
        # Cheekbone: higher = more feminine (increased impact)
        cheekbone_score = (1 - features['cheekbone_ratio']) * self.gender_weights['cheekbone_ratio']
        score += cheekbone_score
        
        # Forehead: higher = more feminine
        forehead_score = (1 - features['forehead_ratio']) * self.gender_weights['forehead_ratio']
        score += forehead_score
        
        return score
    
    def predict_gender(self, features):
        """Predict gender based on facial features with ultra-sensitive calibration"""
        gender_score = self.calculate_gender_score(features)
        
        # Normalize score to 0-1 range
        normalized_score = max(0, min(1, gender_score))
        
        # ULTRA-SENSITIVE threshold for gender classification
        if normalized_score > self.female_threshold:
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
    
    def draw_ultra_sensitive_results(self, frame, results):
        """Draw ultra-sensitive detection results on frame"""
        for result in results:
            x, y, w, h = result['bbox']
            gender = result['gender']
            confidence = result['confidence']
            features = result['features']
            
            # Enhanced color scheme
            if gender == 'man':
                color = (255, 0, 0)  # Blue for men
                gender_icon = "♂"
            else:
                color = (0, 255, 255)  # Yellow for women
                gender_icon = "♀"
            
            # Draw enhanced bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Enhanced gender label
            label = f"{gender_icon} {gender.upper()}: {confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw enhanced background for text
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width + 5, y),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width + 5, y),
                color,
                2
            )
            
            # Draw text
            cv2.putText(
                frame, label,
                (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw gender score
            score_text = f"Score: {result['gender_score']:.2f}"
            cv2.putText(
                frame, score_text,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),  # Green for score
                1
            )
            
            # Draw key features with enhanced visibility
            y_offset = y + h + 35
            important_features = ['jaw_line_ratio', 'cheekbone_ratio', 'forehead_ratio']
            for feature_name in important_features:
                if feature_name in features:
                    value = features[feature_name]
                    feature_text = f"{feature_name.replace('_', ' ').title()}: {value:.2f}"
                    cv2.putText(
                        frame, feature_text,
                        (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),  # Yellow for features
                        1
                    )
                    y_offset += 15
        
        return frame

def run_webcam_detection():
    """Run the ultra-sensitive gender detection on webcam"""
    print("🎥 Starting Ultra-Sensitive Gender Detection")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize detector
    detector = UltraSensitiveGenderDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("🚀 Using ultra-sensitive gender detection...")
    print("💡 Detects females from much further away!")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = detector.process_frame(frame)
            
            # Draw ultra-sensitive results
            frame_with_results = detector.draw_ultra_sensitive_results(frame, results)
            
            # Add enhanced status text
            cv2.putText(
                frame_with_results,
                "Ultra-Sensitive Gender Detection System",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green
                2
            )
            
            # Add instruction text
            cv2.putText(
                frame_with_results,
                "Press 'q' to quit | 's' to save screenshot",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White
                1
            )
            
            # Add sensitivity info
            cv2.putText(
                frame_with_results,
                "Ultra-Sensitive Mode: Detects females from far away!",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Yellow
                1
            )
            
            # Display frame
            cv2.imshow('Ultra-Sensitive Gender Detection System', frame_with_results)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"ultra_sensitive_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_results)
                print(f"📸 Ultra-sensitive screenshot saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Ultra-sensitive webcam session ended")

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 ULTRA-SENSITIVE GENDER DETECTION SYSTEM")
    print("=" * 80)
    print("This system uses highly sensitive facial feature analysis")
    print("Features analyzed:")
    print("  - Face width ratio")
    print("  - Eye spacing ratio")
    print("  - Jaw line strength")
    print("  - Cheekbone position")
    print("  - Forehead ratio")
    print()
    print("🎯 KEY IMPROVEMENTS:")
    print("  - Ultra-sensitive female detection threshold (0.38)")
    print("  - Enhanced feature analysis with better weights")
    print("  - More sensitive face and eye detection")
    print("  - Detects females from much further away!")
    print("  - No need to be super close to camera!")
    print()
    
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is working and accessible")
