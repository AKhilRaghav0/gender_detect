#!/usr/bin/env python3
"""
Logic-Based Gender Detection System
Uses facial features and characteristics to determine gender
This will be used as a baseline before training the neural network
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
import dlib
import math

class LogicBasedGenderDetector:
    def __init__(self):
        """Initialize the logic-based gender detector"""
        
        # Initialize dlib's face detector and facial landmark predictor
        try:
            # Try to load pre-trained models
            self.face_detector = dlib.get_frontal_face_detector()
            
            # Download facial landmark predictor if not present
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(predictor_path):
                print("📥 Downloading facial landmark predictor...")
                self.download_landmark_predictor()
            
            self.landmark_predictor = dlib.shape_predictor(predictor_path)
            print("✅ Dlib models loaded successfully")
            
        except Exception as e:
            print(f"❌ Dlib initialization failed: {e}")
            print("🔄 Falling back to OpenCV-based detection")
            self.use_dlib = False
            self.setup_opencv_detector()
        
        # Gender detection parameters
        self.gender_weights = {
            'jaw_width_ratio': 0.25,      # Wider jaw = more masculine
            'brow_height_ratio': 0.20,    # Higher brows = more feminine
            'lip_thickness_ratio': 0.15,  # Fuller lips = more feminine
            'cheekbone_ratio': 0.20,      # Higher cheekbones = more feminine
            'nose_width_ratio': 0.20      # Narrower nose = more feminine
        }
        
        print("🧠 Logic-based gender detection system initialized")
        print("📊 Using facial feature analysis for gender prediction")
    
    def download_landmark_predictor(self):
        """Download the facial landmark predictor model"""
        import urllib.request
        
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        filename = "shape_predictor_68_face_landmarks.dat.bz2"
        
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✅ Downloaded {filename}")
            
            # Extract the bz2 file
            import bz2
            with bz2.open(filename, 'rb') as source, open('shape_predictor_68_face_landmarks.dat', 'wb') as target:
                target.write(source.read())
            
            # Clean up
            os.remove(filename)
            print("✅ Extracted facial landmark predictor")
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise
    
    def setup_opencv_detector(self):
        """Setup OpenCV-based face detection as fallback"""
        self.use_dlib = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ OpenCV face detector loaded as fallback")
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        if hasattr(self, 'use_dlib') and self.use_dlib:
            # Use dlib for better accuracy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            return [(face.left(), face.top(), face.width(), face.height()) for face in faces]
        else:
            # Use OpenCV as fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def extract_facial_features(self, frame, face_bbox):
        """Extract facial features for gender analysis"""
        x, y, w, h = face_bbox
        
        if hasattr(self, 'use_dlib') and self.use_dlib:
            # Use dlib for precise landmark detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = self.landmark_predictor(gray, face_rect)
            
            # Extract key facial measurements
            features = self.analyze_dlib_landmarks(landmarks, w, h)
        else:
            # Use OpenCV-based feature extraction
            features = self.analyze_opencv_features(frame, face_bbox)
        
        return features
    
    def analyze_dlib_landmarks(self, landmarks, face_width, face_height):
        """Analyze facial features using dlib landmarks"""
        features = {}
        
        # Get landmark points
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Jaw width ratio (landmarks 0-16)
        jaw_width = np.linalg.norm(points[0] - points[16])
        features['jaw_width_ratio'] = jaw_width / face_width
        
        # Brow height ratio (landmarks 17-21, 22-26)
        left_brow_height = np.mean([points[i][1] for i in range(17, 22)])
        right_brow_height = np.mean([points[i][1] for i in range(22, 27)])
        brow_height = (left_brow_height + right_brow_height) / 2
        features['brow_height_ratio'] = (face_height - brow_height) / face_height
        
        # Lip thickness ratio (landmarks 48-67)
        lip_height = np.linalg.norm(points[51] - points[57])
        features['lip_thickness_ratio'] = lip_height / face_height
        
        # Cheekbone ratio (landmarks 1, 15)
        cheekbone_width = np.linalg.norm(points[1] - points[15])
        features['cheekbone_ratio'] = cheekbone_width / face_width
        
        # Nose width ratio (landmarks 31-35)
        nose_width = np.linalg.norm(points[31] - points[35])
        features['nose_width_ratio'] = nose_width / face_width
        
        return features
    
    def analyze_opencv_features(self, frame, face_bbox):
        """Analyze facial features using OpenCV"""
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        
        features = {}
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Jaw width ratio (approximate using face width)
        features['jaw_width_ratio'] = 0.8  # Default value
        
        # Brow height ratio (approximate using upper third of face)
        features['brow_height_ratio'] = 0.3  # Default value
        
        # Lip thickness ratio (detect lips in lower third)
        lip_region = gray_face[int(h*0.6):h, :]
        if lip_region.size > 0:
            # Simple edge detection for lips
            edges = cv2.Canny(lip_region, 50, 150)
            lip_density = np.sum(edges > 0) / edges.size
            features['lip_thickness_ratio'] = min(lip_density * 2, 0.3)
        else:
            features['lip_thickness_ratio'] = 0.15
        
        # Cheekbone ratio (approximate)
        features['cheekbone_ratio'] = 0.7  # Default value
        
        # Nose width ratio (approximate)
        features['nose_width_ratio'] = 0.25  # Default value
        
        return features
    
    def calculate_gender_score(self, features):
        """Calculate gender score based on facial features"""
        score = 0.0
        
        # Jaw width: wider = more masculine
        jaw_score = features['jaw_width_ratio'] * self.gender_weights['jaw_width_ratio']
        score += jaw_score
        
        # Brow height: higher = more feminine
        brow_score = (1 - features['brow_height_ratio']) * self.gender_weights['brow_height_ratio']
        score += brow_score
        
        # Lip thickness: fuller = more feminine
        lip_score = features['lip_thickness_ratio'] * self.gender_weights['lip_thickness_ratio']
        score += lip_score
        
        # Cheekbone: higher = more feminine
        cheekbone_score = (1 - features['cheekbone_ratio']) * self.gender_weights['cheekbone_ratio']
        score += cheekbone_score
        
        # Nose width: narrower = more feminine
        nose_score = (1 - features['nose_width_ratio']) * self.gender_weights['nose_width_ratio']
        score += nose_score
        
        return score
    
    def predict_gender(self, features):
        """Predict gender based on facial features"""
        gender_score = self.calculate_gender_score(features)
        
        # Normalize score to 0-1 range
        normalized_score = max(0, min(1, gender_score))
        
        # Threshold for gender classification
        if normalized_score > 0.6:
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
            feature_text = f"Jaw: {features['jaw_width_ratio']:.2f}"
            cv2.putText(
                frame, feature_text,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        
        return frame

def run_webcam_detection():
    """Run the logic-based gender detection on webcam"""
    print("🎥 Starting Logic-Based Gender Detection")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Initialize detector
    detector = LogicBasedGenderDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera opened successfully!")
    print("🧠 Using logic-based gender detection...")
    
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
            cv2.imshow('Logic-Based Gender Detection', frame_with_results)
            
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
    print("🧠 Logic-Based Gender Detection System")
    print("=" * 70)
    print("This system uses facial feature analysis to determine gender")
    print("Features analyzed:")
    print("  - Jaw width ratio")
    print("  - Brow height ratio") 
    print("  - Lip thickness ratio")
    print("  - Cheekbone ratio")
    print("  - Nose width ratio")
    print()
    
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try installing dlib: pip install dlib")
