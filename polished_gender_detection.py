#!/usr/bin/env python3
"""
Polished Gender Detection System
Clean, minimal UI with enhanced detection and polished interface
Professional tool for demonstrations with sleek design
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

class PolishedGenderDetector:
    def __init__(self):
        """Initialize the polished gender detector"""
        
        # Load face detection models
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load additional cascades for better feature detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # POLISHED gender detection parameters (optimized)
        self.gender_weights = {
            'face_width_ratio': 0.11,      # Wider face = more masculine
            'eye_spacing_ratio': 0.17,     # Wider eye spacing = more masculine
            'jaw_line_ratio': 0.33,        # Stronger jaw = more masculine
            'cheekbone_ratio': 0.23,       # Higher cheekbones = more feminine
            'forehead_ratio': 0.16         # Higher forehead = more feminine
        }
        
        # OPTIMAL threshold for both genders
        self.female_threshold = 0.40
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.detection_times = []
        
        print("✨ Polished Gender Detection System")
        print("🎨 Clean, minimal UI with enhanced detection")
        print("💎 Professional tool for demonstrations")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.06, 3, minSize=(42, 42))
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_eyes(self, face_roi):
        """Detect eyes within a face region"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.06, 3, minSize=(14, 14))
        return eyes
    
    def analyze_facial_features(self, frame, face_bbox):
        """Analyze facial features for gender determination"""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        features = {}
        
        # 1. Face width ratio
        features['face_width_ratio'] = w / h
        
        # 2. Eye spacing ratio
        eyes = self.detect_eyes(face_roi)
        if len(eyes) >= 2:
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_center = (ex + ew//2, ey + eh//2)
                eye_centers.append(eye_center)
            
            if len(eye_centers) >= 2:
                eye_centers.sort(key=lambda p: p[0])
                eye_spacing = np.linalg.norm(np.array(eye_centers[1]) - np.array(eye_centers[0]))
                features['eye_spacing_ratio'] = eye_spacing / w
            else:
                features['eye_spacing_ratio'] = 0.305
        else:
            features['eye_spacing_ratio'] = 0.305
        
        # 3. Jaw line strength
        lower_face = face_roi[int(h*0.6):h, :]
        if lower_face.size > 0:
            gray_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_lower, 22, 72)
            jaw_strength = np.sum(edges > 0) / edges.size
            features['jaw_line_ratio'] = min(jaw_strength * 2.15, 1.0)
        else:
            features['jaw_line_ratio'] = 0.5
        
        # 4. Cheekbone position
        middle_face = face_roi[int(h*0.3):int(h*0.7), :]
        if middle_face.size > 0:
            gray_middle = cv2.cvtColor(middle_face, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray_middle, cv2.CV_64F, 1, 0, ksize=3)
            cheekbone_strength = np.sum(np.abs(sobel_x) > 31) / sobel_x.size
            features['cheekbone_ratio'] = min(cheekbone_strength * 1.7, 1.0)
        else:
            features['cheekbone_ratio'] = 0.5
        
        # 5. Forehead ratio
        upper_face = face_roi[:int(h*0.4), :]
        if upper_face.size > 0:
            gray_upper = cv2.cvtColor(upper_face, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray_upper, cv2.CV_64F)
            smoothness = 1.0 - (np.sum(np.abs(laplacian) > 19) / laplacian.size)
            features['forehead_ratio'] = smoothness
        else:
            features['forehead_ratio'] = 0.5
        
        return features
    
    def calculate_gender_score(self, features):
        """Calculate gender score based on facial features"""
        score = 0.0
        
        face_width_score = features['face_width_ratio'] * self.gender_weights['face_width_ratio']
        score += face_width_score
        
        eye_spacing_score = features['eye_spacing_ratio'] * self.gender_weights['eye_spacing_ratio']
        score += eye_spacing_score
        
        jaw_score = features['jaw_line_ratio'] * self.gender_weights['jaw_line_ratio']
        score += jaw_score
        
        cheekbone_score = (1 - features['cheekbone_ratio']) * self.gender_weights['cheekbone_ratio']
        score += cheekbone_score
        
        forehead_score = (1 - features['forehead_ratio']) * self.gender_weights['forehead_ratio']
        score += forehead_score
        
        return score
    
    def predict_gender(self, features):
        """Predict gender based on facial features"""
        gender_score = self.calculate_gender_score(features)
        normalized_score = max(0, min(1, gender_score))
        
        if normalized_score > self.female_threshold:
            gender = 'woman'
            confidence = normalized_score
        else:
            gender = 'man'
            confidence = 1 - normalized_score
        
        return gender, confidence, features
    
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        start_time = time.time()
        
        faces = self.detect_faces(frame)
        results = []
        
        for face_bbox in faces:
            try:
                features = self.analyze_facial_features(frame, face_bbox)
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
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 30:  # Keep last 30 frames
            self.detection_times.pop(0)
        
        return results
    
    def draw_polished_results(self, frame, results):
        """Draw polished detection results with enhanced UI"""
        for result in results:
            x, y, w, h = result['bbox']
            gender = result['gender']
            confidence = result['confidence']
            features = result['features']
            
            # Polished color scheme
            if gender == 'man':
                primary_color = (0, 120, 255)      # Orange
                secondary_color = (0, 80, 200)     # Darker orange
                accent_color = (0, 200, 255)       # Light orange
                gender_icon = "♂"
            else:
                primary_color = (255, 0, 255)      # Magenta
                secondary_color = (200, 0, 200)    # Darker magenta
                accent_color = (255, 100, 255)     # Light magenta
                gender_icon = "♀"
            
            # Enhanced face detection box with multiple layers
            # 1. Outer shadow/glow effect
            for i in range(3, 0, -1):
                alpha = 0.3 - (i * 0.1)
                shadow_color = (int(primary_color[0] * alpha), 
                              int(primary_color[1] * alpha), 
                              int(primary_color[2] * alpha))
                cv2.rectangle(frame, (x-i, y-i), (x + w+i, y + h+i), shadow_color, 1)
            
            # 2. Main detection box with thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), primary_color, 3)
            
            # 3. Inner highlight box
            cv2.rectangle(frame, (x+2, y+2), (x + w-2, y + h-2), accent_color, 1)
            
            # 4. Corner indicators for precision
            corner_size = 15
            # Top-left corner
            cv2.line(frame, (x, y), (x + corner_size, y), primary_color, 2)
            cv2.line(frame, (x, y), (x, y + corner_size), primary_color, 2)
            # Top-right corner
            cv2.line(frame, (x + w - corner_size, y), (x + w, y), primary_color, 2)
            cv2.line(frame, (x + w, y), (x + w, y + corner_size), primary_color, 2)
            # Bottom-left corner
            cv2.line(frame, (x, y + h - corner_size), (x, y + h), primary_color, 2)
            cv2.line(frame, (x, y + h), (x + corner_size, y + h), primary_color, 2)
            # Bottom-right corner
            cv2.line(frame, (x + w - corner_size, y + h), (x + w, y + h), primary_color, 2)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), primary_color, 2)
            
            # Polished gender label with clean design
            label = f"{gender_icon} {gender.upper()}"
            confidence_text = f"{confidence:.1%}"
            
            # Get text sizes
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
            )
            (conf_width, conf_height), _ = cv2.getTextSize(
                confidence_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1
            )
            
            # Draw clean label background
            label_y = y - 15
            cv2.rectangle(
                frame,
                (x, label_y - label_height - 8),
                (x + label_width + 20, label_y + 5),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                frame,
                (x, label_y - label_height - 8),
                (x + label_width + 20, label_y + 5),
                primary_color,
                2
            )
            
            # Draw label text
            cv2.putText(
                frame, label,
                (x + 10, label_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw confidence below label
            conf_y = label_y + conf_height + 5
            cv2.putText(
                frame, confidence_text,
                (x + 10, conf_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                accent_color,
                1
            )
            
            # Draw clean analysis panel
            panel_x = x + w + 15
            panel_y = y
            panel_width = 200
            panel_height = 120
            
            # Panel background with clean style
            cv2.rectangle(
                frame,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                (0, 0, 0),
                -1
            )
            cv2.rectangle(
                frame,
                (panel_x, panel_y),
                (panel_x + panel_width, panel_y + panel_height),
                primary_color,
                2
            )
            
            # Panel title
            cv2.putText(
                frame, "ANALYSIS",
                (panel_x + 10, panel_y + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            # Feature values with clean precision
            y_offset = panel_y + 40
            important_features = ['jaw_line_ratio', 'cheekbone_ratio', 'forehead_ratio']
            for feature_name in important_features:
                if feature_name in features:
                    value = features[feature_name]
                    feature_text = f"{feature_name.replace('_', ' ').title()}: {value:.2f}"
                    cv2.putText(
                        frame, feature_text,
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        accent_color,
                        1
                    )
                    y_offset += 15
            
            # Overall score
            score_text = f"Score: {result['gender_score']:.2f}"
            cv2.putText(
                frame, score_text,
                (panel_x + 10, y_offset + 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return frame
    
    def draw_minimal_header(self, frame):
        """Draw minimal header with essential info only"""
        # Minimal header background
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 255, 255), 1)
        
        # Simple title
        cv2.putText(
            frame, "Gender Detection System",
            (20, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 255, 255),
            1
        )
        
        # Minimal metrics
        metrics_text = f"FPS: {self.fps:.1f} | Faces: {len(self.get_current_faces())}"
        cv2.putText(
            frame, metrics_text,
            (frame.shape[1] - 200, 25),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (150, 255, 150),
            1
        )
    
    def draw_minimal_footer(self, frame):
        """Draw minimal footer with essential info"""
        footer_y = frame.shape[0] - 25
        
        # Footer background
        cv2.rectangle(frame, (0, footer_y), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, footer_y), (frame.shape[1], frame.shape[0]), (0, 255, 255), 1)
        
        # Simple instructions
        instructions = "Press 'q' to quit | 's' to save | 'h' for help"
        cv2.putText(
            frame, instructions,
            (20, footer_y + 18),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (200, 200, 200),
            1
        )
        
        # Simple system info
        system_info = f"Threshold: {self.female_threshold} | Features: 5"
        cv2.putText(
            frame, system_info,
            (frame.shape[1] - 250, footer_y + 18),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (200, 200, 200),
            1
        )
    
    def get_current_faces(self):
        """Get current detected faces for metrics"""
        return []  # This will be updated in the main loop
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = current_time

def run_polished_webcam_detection():
    """Run the polished gender detection on webcam"""
    print("✨ Starting Polished Gender Detection System")
    print("🎨 Clean, minimal UI with enhanced detection")
    print("💎 Professional tool for demonstrations")
    print("Press 'q' to quit, 's' to save screenshot, 'h' for help")
    
    # Initialize detector
    detector = PolishedGenderDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    # Set camera properties for quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized!")
    print("🎨 Using polished UI with clean design...")
    print("💎 Ready for demonstration!")
    
    current_faces = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update FPS
            detector.update_fps()
            
            # Process frame
            results = detector.process_frame(frame)
            current_faces = results
            
            # Draw polished results
            frame_with_results = detector.draw_polished_results(frame, results)
            
            # Draw minimal header and footer
            detector.draw_minimal_header(frame_with_results)
            detector.draw_minimal_footer(frame_with_results)
            
            # Display frame
            cv2.imshow('✨ Polished Gender Detection System', frame_with_results)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"polished_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_results)
                print(f"📸 Polished screenshot saved: {filename}")
            elif key == ord('h'):
                print("💡 Help:")
                print("   - Press 'q' to quit")
                print("   - Press 's' to save screenshot")
                print("   - Press 'h' for this help")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Polished webcam session ended")

if __name__ == "__main__":
    print("=" * 80)
    print("✨ POLISHED GENDER DETECTION SYSTEM")
    print("=" * 80)
    print("Clean, minimal UI with enhanced detection")
    print("Professional tool for demonstrations")
    print()
    print("🎨 FEATURES:")
    print("  - Clean, minimal interface")
    print("  - Enhanced face detection boxes")
    print("  - Polished visual elements")
    print("  - Professional appearance")
    print()
    
    try:
        run_polished_webcam_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is working and accessible")
