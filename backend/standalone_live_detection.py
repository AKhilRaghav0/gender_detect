"""
Standalone Live Webcam Gender Detection
Uses the same logic as polished_gender_detection.py
No backend dependencies - for testing only
"""

import cv2
import numpy as np
import time
from datetime import datetime

class StandaloneLiveDetection:
    def __init__(self):
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Gender detection parameters (same as polished_gender_detection.py)
        self.female_threshold = 0.40
        self.gender_weights = {
            'face_width': 0.25,
            'eye_spacing': 0.20,
            'jaw_strength': 0.25,
            'cheekbone_position': 0.15,
            'forehead_ratio': 0.15
        }
        
        # Frame processing settings
        self.frame_interval = 4  # Process every 4 seconds
        self.last_frame_time = 0
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open webcam!")
        
        print("🎥 Webcam initialized successfully!")
        print("💡 Press 'q' to quit, 's' to save frame")
    
    def analyze_facial_features(self, face_roi, eyes, face_width, face_height):
        """Analyze facial features for gender prediction (same logic as polished)"""
        try:
            # Convert to grayscale for analysis
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate facial feature ratios
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
            # Use edge detection to measure jaw definition
            edges = cv2.Canny(gray_face, 50, 150)
            jaw_region = edges[face_height//2:, :]
            features['jaw_strength'] = np.sum(jaw_region > 0) / (face_width * face_height//2)
            
            # Cheekbone position (higher cheekbones tend to be female)
            cheek_region = gray_face[face_height//4:3*face_height//4, :]
            features['cheekbone_position'] = np.mean(cheek_region) / 255.0
            
            # Forehead ratio (larger forehead tends to be male)
            forehead_region = gray_face[:face_height//4, :]
            features['forehead_ratio'] = np.mean(forehead_region) / 255.0
            
            # Calculate weighted gender score
            gender_score = 0.0
            for feature, weight in self.gender_weights.items():
                if feature in features:
                    # Normalize feature values and apply weights
                    if feature in ['face_width', 'eye_spacing', 'jaw_strength']:
                        gender_score += features[feature] * weight
                    else:  # cheekbone_position, forehead_ratio
                        gender_score += (1.0 - features[feature]) * weight
            
            return gender_score
            
        except Exception as e:
            print(f"Error in facial analysis: {e}")
            return 0.5
    
    def detect_gender(self, frame):
        """Detect faces and predict gender"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        confidence_scores = {}
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(
                gray[y:y+h, x:x+w], scaleFactor=1.1, minNeighbors=3, minSize=(15, 15)
            )
            
            # Analyze facial features
            gender_score = self.analyze_facial_features(face_roi, eyes, w, h)
            
            # Predict gender based on threshold
            if gender_score > self.female_threshold:
                gender = 'female'
                confidence = gender_score
                color = (255, 0, 255)  # Magenta for female
            else:
                gender = 'male'
                confidence = 1.0 - gender_score
                color = (0, 165, 255)  # Orange for male
            
            gender_counts[gender] += 1
            confidence_scores[gender] = confidence
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add gender label
            label = f"{gender.title()}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return len(faces), gender_counts, confidence_scores
    
    def draw_ui(self, frame, face_count, gender_counts, processing_time):
        """Draw user interface on frame"""
        # Header
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Live Gender Detection - Bus Safety System", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Live Feed - {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Results panel
        panel_y = 80
        cv2.rectangle(frame, (10, panel_y), (300, panel_y + 120), (0, 0, 0), -1)
        
        # Face count
        cv2.putText(frame, f"Faces Detected: {face_count}", (20, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Gender counts with colors
        y_offset = 50
        for gender, count in gender_counts.items():
            if count > 0:
                color = (255, 0, 255) if gender == 'female' else (0, 165, 255) if gender == 'male' else (128, 128, 128)
                cv2.putText(frame, f"{gender.title()}: {count}", (20, panel_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        # Processing info
        cv2.putText(frame, f"Processing: {processing_time:.2f}s", (20, panel_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame interval indicator
        time_since_last = time.time() - self.last_frame_time
        if time_since_last < self.frame_interval:
            remaining = self.frame_interval - time_since_last
            cv2.putText(frame, f"Next frame in: {remaining:.1f}s", (320, panel_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Processing frame...", (320, panel_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Footer
        cv2.rectangle(frame, (0, 420), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Frame interval: 4s | Memory optimized", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main detection loop"""
        print("🚀 Starting standalone live detection...")
        
        # Initialize variables
        face_count = 0
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        processing_time = 0.0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                current_time = time.time()
                
                # Process frame every 4 seconds to save memory
                if current_time - self.last_frame_time >= self.frame_interval:
                    print("🔍 Processing new frame...")
                    
                    start_time = time.time()
                    face_count, gender_counts, confidence_scores = self.detect_gender(frame)
                    processing_time = time.time() - start_time
                    
                    self.last_frame_time = current_time
                    print(f"✅ Processed {face_count} faces in {processing_time:.2f}s")
                    print(f"📊 Results: {gender_counts}")
                
                # Draw UI on frame
                self.draw_ui(frame, face_count, gender_counts, processing_time)
                
                # Display frame
                cv2.imshow('Standalone Live Gender Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"standalone_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved frame as {filename}")
                
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
    print("🚌 Standalone Live Webcam Gender Detection")
    print("=" * 50)
    print("🎯 Uses the same logic as polished_gender_detection.py")
    print("💾 Processes frames every 4 seconds to save memory")
    print("=" * 50)
    
    try:
        detector = StandaloneLiveDetection()
        detector.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
