"""
Live Polished Gender Detection System
Exact same UI and logic as polished_gender_detection.py
But with real-time live webcam processing
"""

import cv2
import numpy as np
import time
from datetime import datetime

class LivePolishedDetection:
    def __init__(self):
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Gender detection parameters (EXACT same as polished_gender_detection.py)
        self.female_threshold = 0.55  # Increased from 0.40 to 0.55 for better accuracy
        self.gender_weights = {
            'face_width': 0.25,
            'eye_spacing': 0.20,
            'jaw_strength': 0.25,
            'cheekbone_position': 0.15,
            'forehead_ratio': 0.15
        }
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open webcam!")
        
        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0.0
        
        print("🎥 Webcam initialized successfully!")
        print("💡 Press 'q' to quit, 's' to save frame")
        print("✨ Using EXACT same UI as polished_gender_detection.py")
    
    def analyze_facial_features(self, face_roi, eyes, face_width, face_height):
        """Analyze facial features for gender prediction (EXACT same as polished)"""
        try:
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate facial feature ratios (EXACT same logic)
            features = {}
            
            # Face width ratio (broader faces tend to be male)
            features['face_width'] = min(face_width / face_height, 1.0)
            
            # Eye spacing (wider spacing tends to be male)
            if len(eyes) >= 2:
                eye_centers = [(e[0] + e[2]//2, e[1] + e[3]//2) for e in eyes]
                eye_distance = np.sqrt((eye_centers[1][0] - eye_centers[0][0])**2 + 
                                     (eye_centers[1][1] - eye_centers[0][1])**2)
                features['eye_spacing'] = min(eye_distance / face_width, 1.0)
            else:
                features['eye_spacing'] = 0.5
            
            # Jaw line strength (stronger jaw tends to be male)
            edges = cv2.Canny(gray_face, 50, 150)
            jaw_region = edges[face_height//2:, :]
            features['jaw_strength'] = np.sum(jaw_region > 0) / (face_width * face_height//2)
            
            # Cheekbone position (higher cheekbones tend to be female)
            cheek_region = gray_face[face_height//4:3*face_height//4, :]
            features['cheekbone_position'] = np.mean(cheek_region) / 255.0
            
            # Forehead ratio (larger forehead tends to be male)
            forehead_region = gray_face[:face_height//4, :]
            features['forehead_ratio'] = np.mean(forehead_region) / 255.0
            
            # Calculate weighted gender score (EXACT same formula)
            gender_score = 0.0
            for feature, weight in self.gender_weights.items():
                if feature in features:
                    if feature in ['face_width', 'eye_spacing', 'jaw_strength']:
                        gender_score += features[feature] * weight
                    else:  # cheekbone_position, forehead_ratio
                        gender_score += (1.0 - features[feature]) * weight
            
            return gender_score, features
            
        except Exception as e:
            print(f"Error in facial analysis: {e}")
            return 0.5, {}
    
    def detect_faces(self, frame):
        """Detect faces (EXACT same parameters as polished)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.06, 3, minSize=(42, 42))
        return faces
    
    def detect_eyes(self, face_roi):
        """Detect eyes (EXACT same parameters as polished)"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3, minSize=(15, 15))
        return eyes
    
    def process_frame(self, frame):
        """Process frame for gender detection (EXACT same logic as polished)"""
        faces = self.detect_faces(frame)
        face_count = len(faces)
        
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        confidence_scores = {}
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            eyes = self.detect_eyes(face_roi)
            
            # Analyze facial features
            gender_score, features = self.analyze_facial_features(face_roi, eyes, w, h)
            
            # Predict gender based on threshold (EXACT same logic)
            if gender_score > self.female_threshold:
                gender = 'female'
                confidence = gender_score
                color = (255, 0, 255)  # Magenta for female
                print(f"🔍 DEBUG: Score {gender_score:.3f} > Threshold {self.female_threshold} → FEMALE")
            else:
                gender = 'male'
                confidence = 1.0 - gender_score
                color = (0, 165, 255)  # Orange for male
                print(f"🔍 DEBUG: Score {gender_score:.3f} ≤ Threshold {self.female_threshold} → MALE")
            
            gender_counts[gender] += 1
            confidence_scores[gender] = confidence
            
            # Draw EXACT same bounding box style as polished
            self.draw_polished_face_box(frame, x, y, w, h, gender, confidence, features, color)
        
        return face_count, gender_counts, confidence_scores
    
    def draw_polished_face_box(self, frame, x, y, w, h, gender, confidence, features, color):
        """Draw EXACT same face box style as polished_gender_detection.py"""
        # Multi-layer bounding box (EXACT same as polished)
        box_thickness = 2
        
        # Outer glow effect
        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, box_thickness)
        
        # Main bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, box_thickness)
        
        # Inner highlight
        cv2.rectangle(frame, (x+1, y+1), (x+w-1, y+h-1), (255, 255, 255), 1)
        
        # Corner indicators (EXACT same as polished)
        corner_size = 8
        # Top-left corner
        cv2.line(frame, (x, y), (x+corner_size, y), color, 3)
        cv2.line(frame, (x, y), (x, y+corner_size), color, 3)
        # Top-right corner
        cv2.line(frame, (x+w-corner_size, y), (x+w, y), color, 3)
        cv2.line(frame, (x+w, y), (x+w, y+corner_size), color, 3)
        # Bottom-left corner
        cv2.line(frame, (x, y+h-corner_size), (x, y+h), color, 3)
        cv2.line(frame, (x, y+h), (x+corner_size, y+h), color, 3)
        # Bottom-right corner
        cv2.line(frame, (x+w-corner_size, y+h), (x+w, y+h), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_size), color, 3)
        
        # Gender label (EXACT same style as polished)
        label = f"??? {gender.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Label background
        cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0]+10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence percentage
        confidence_text = f"{confidence*100:.1f}%"
        conf_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.putText(frame, confidence_text, (x+w-conf_size[0]-5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Analysis panel (EXACT same as polished)
        panel_x = x + w + 10
        panel_y = y
        panel_width = 200
        panel_height = 120
        
        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), 
                     color, 2)
        
        # Panel title
        cv2.putText(frame, "ANALYSIS", (panel_x+5, panel_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Feature values (EXACT same as polished)
        y_offset = 40
        for feature, value in features.items():
            if feature in ['jaw_strength', 'cheekbone_position', 'forehead_ratio']:
                feature_text = f"{feature.replace('_', ' ').title()}: {value:.2f}"
                cv2.putText(frame, feature_text, (panel_x+5, panel_y+y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15
        
        # Final score
        cv2.putText(frame, f"Score: {confidence:.2f}", (panel_x+5, panel_y+y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_polished_ui(self, frame, face_count, gender_counts):
        """Draw EXACT same UI as polished_gender_detection.py"""
        # Header (EXACT same style)
        header_height = 40
        cv2.rectangle(frame, (0, 0), (640, header_height), (0, 0, 0), -1)
        cv2.putText(frame, "Gender Detection System", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Top-right info (EXACT same as polished)
        # FPS calculation
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        fps_text = f"FPS: {self.fps:.1f}"
        faces_text = f"Faces: {face_count}"
        
        cv2.putText(frame, fps_text, (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, faces_text, (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Footer (EXACT same as polished)
        footer_height = 25
        cv2.rectangle(frame, (0, 480-footer_height), (640, 480), (0, 0, 0), -1)
        
        # Left side instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save | 'h' for help", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Right side info (EXACT same as polished)
        threshold_text = f"Threshold: {self.female_threshold}"
        features_text = f"Features: {len(self.gender_weights)}"
        
        cv2.putText(frame, threshold_text, (400, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, features_text, (520, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main detection loop with real-time processing"""
        print("🚀 Starting Live Polished Gender Detection...")
        print("✨ EXACT same UI as polished_gender_detection.py")
        print("⚡ Real-time processing (no delays)")
        
        # Initialize variables
        face_count = 0
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Process frame in REAL-TIME (no delays)
                start_time = time.time()
                face_count, gender_counts, confidence_scores = self.process_frame(frame)
                processing_time = time.time() - start_time
                
                # Draw EXACT same UI as polished
                self.draw_polished_ui(frame, face_count, gender_counts)
                
                # Display frame
                cv2.imshow('Live Polished Gender Detection System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"live_polished_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved frame as {filename}")
                elif key == ord('h'):
                    print("💡 Help: Press 'q' to quit, 's' to save frame")
                
        except KeyboardInterrupt:
            print("🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup completed")

def main():
    """Main function"""
    print("🚌 Live Polished Gender Detection System")
    print("=" * 60)
    print("✨ EXACT same UI as polished_gender_detection.py")
    print("⚡ Real-time processing (no delays)")
    print("🎨 Professional polished interface")
    print("=" * 60)
    
    try:
        detector = LivePolishedDetection()
        detector.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
