"""
Advanced Live Gender Detection System
Combines SCRFD face detection with deep learning gender classification
"""

import cv2
import numpy as np
import time
from datetime import datetime
from scrfd_detection import create_scrfd_detector
from gender_classifier import create_gender_classifier

class LiveAdvancedGenderDetection:
    def __init__(self):
        # Initialize SCRFD detector
        self.face_detector = create_scrfd_detector(conf_threshold=0.5)

        # Initialize advanced gender classifier
        self.gender_classifier = create_gender_classifier(device='auto')

        # Gender detection parameters (for fallback)
        self.female_threshold = 0.55
        self.gender_weights = {
            'face_width': 0.25,
            'eye_spacing': 0.20,
            'jaw_strength': 0.25,
            'cheekbone_position': 0.15,
            'forehead_ratio': 0.15
        }

        # Initialize camera with enhanced compatibility
        self.cap = None
        self._init_camera_advanced()

    def _init_camera_advanced(self):
        """Initialize camera with advanced compatibility checking"""
        print("📹 Initializing advanced camera system...")

        # Detect environment
        import platform
        is_wsl = 'microsoft' in platform.release().lower() or 'wsl' in platform.release().lower()

        if is_wsl:
            print("🐧 WSL detected - camera access may be limited")
            print("💡 Tip: Consider using Windows Python for better camera support")

        # Try different camera sources
        sources = [
            ("Direct Camera 0", 0),
            ("Direct Camera 1", 1),
            ("Direct Camera 2", 2),
        ]

        # Add IP camera options for WSL
        if is_wsl:
            ip_options = [
                "http://192.168.1.100:8080/video",
                "http://10.0.0.100:8080/video",
            ]
            for ip_url in ip_options:
                sources.append((f"IP Camera ({ip_url})", ip_url))

        for name, source in sources:
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        print(f"✅ {name}: {test_frame.shape[1]}x{test_frame.shape[0]} resolution")
                        return
                cap.release()
            except Exception as e:
                print(f"⚠️ {name}: Failed - {e}")

        # If no camera found, show help
        self._show_advanced_camera_help(is_wsl)

    def _show_advanced_camera_help(self, is_wsl):
        """Show advanced camera troubleshooting"""
        help_msg = f"""
❌ Cannot initialize camera!

🔍 ENVIRONMENT: {'WSL (Limited camera support)' if is_wsl else 'Native Windows/Linux'}

📋 SOLUTIONS:
"""

        if is_wsl:
            help_msg += """
🐧 WSL SOLUTIONS:
1. 🎯 BEST: Use Windows Python instead
   cd /mnt/c/Users/YourName/path/to/project
   python backend/live_advanced_gender_detection.py

2. 📱 IP Webcam (Recommended for WSL):
   • Install IP Webcam app on phone
   • Start server, get IP: http://192.168.1.xxx:8080
   • The system will auto-detect IP cameras

3. 🔧 Advanced WSL USB (Complex):
   • Requires USBIPD-WIN setup
   • Limited performance
"""

        else:
            help_msg += """
🪟 WINDOWS SOLUTIONS:
1. 📷 Test Windows Camera app first
2. 🔧 Check camera permissions in Settings
3. 🔄 Try different USB ports
4. 🔧 Update camera drivers
"""

        help_msg += """
🧪 TEST COMMANDS:
• Camera test: python test_camera_setup.py
• OpenCV test: python -c "import cv2; print('OK')"
• List cameras: python -c "import cv2; [print(f'Camera {i}: OK') for i in range(5) if cv2.VideoCapture(i).isOpened()]"

📞 SUPPORT:
• Run: python test_camera_setup.py (for detailed diagnostics)
• Check: WSL_CAMERA_GUIDE.md (for WSL-specific help)
"""

        raise Exception(help_msg)

        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0.0

        print("🎯 Advanced SCRFD Face Detection initialized!")
        print("🧠 Deep Learning Gender Classification loaded!")
        print("💡 Press 'q' to quit, 's' to save frame, 'b' to benchmark, 'm' to toggle mode")
        print("⚡ Real-time processing with superior accuracy!")

    def analyze_facial_features(self, face_roi, eyes, face_width, face_height):
        """Fallback gender analysis using facial features"""
        try:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

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

            # Calculate weighted gender score
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
        """Detect faces using SCRFD"""
        return self.face_detector.detect_faces(frame)

    def classify_gender_advanced(self, face_roi):
        """Advanced gender classification using deep learning"""
        return self.gender_classifier.classify_gender(face_roi)

    def detect_eyes_in_face(self, face_roi):
        """Detect eyes within a face region"""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Use basic thresholding to find eye regions
        face_height, face_width = face_roi.shape[:2]
        eye_region = gray_face[:face_height//2, :]

        _, thresh = cv2.threshold(eye_region, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        eyes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > face_width//10 and h > face_height//20 and w/h < 3:
                eyes.append((x, y, w, h))

        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        return eyes

    def process_frame(self, frame, use_advanced=True):
        """Process frame for gender detection with advanced classification"""
        faces = self.detect_faces(frame)
        face_count = len(faces)

        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
        confidence_scores = {}

        for face_data in faces:
            x, y, w, h, detection_conf = face_data

            if w <= 0 or h <= 0:
                continue

            x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            if w <= 10 or h <= 10:
                continue

            face_roi = frame[y:y+h, x:x+w]

            # Use advanced gender classification
            if use_advanced:
                gender_result = self.classify_gender_advanced(face_roi)
                gender = gender_result['gender']
                confidence = gender_result['confidence']
                method = "DL"  # Deep Learning
            else:
                # Fallback to feature-based classification
                eyes = self.detect_eyes_in_face(face_roi)
                gender_score, features = self.analyze_facial_features(face_roi, eyes, w, h)

                if gender_score > self.female_threshold:
                    gender = 'female'
                    confidence = gender_score
                else:
                    gender = 'male'
                    confidence = 1.0 - gender_score
                method = "HEUR"  # Heuristic

            # Update counts
            if gender in gender_counts:
                gender_counts[gender] += 1
                confidence_scores[gender] = confidence

            # Choose color based on gender
            if gender == 'female':
                color = (255, 0, 255)  # Magenta
            elif gender == 'male':
                color = (0, 165, 255)  # Orange
            else:
                color = (128, 128, 128)  # Gray for unknown

            # Draw bounding box and info
            self.draw_advanced_face_box(frame, x, y, w, h, gender, confidence, detection_conf, method, color)

        return face_count, gender_counts, confidence_scores

    def draw_advanced_face_box(self, frame, x, y, w, h, gender, confidence, detection_conf, method, color):
        """Draw advanced face box with detailed information"""
        # Multi-layer bounding box
        box_thickness = 2

        # Outer glow effect
        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, box_thickness)

        # Main bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, box_thickness)

        # Inner highlight
        cv2.rectangle(frame, (x+1, y+1), (x+w-1, y+h-1), (255, 255, 255), 1)

        # Corner indicators
        corner_size = 8
        cv2.line(frame, (x, y), (x+corner_size, y), color, 3)
        cv2.line(frame, (x, y), (x, y+corner_size), color, 3)
        cv2.line(frame, (x+w-corner_size, y), (x+w, y), color, 3)
        cv2.line(frame, (x+w, y), (x+w, y+corner_size), color, 3)
        cv2.line(frame, (x, y+h-corner_size), (x, y+h), color, 3)
        cv2.line(frame, (x, y+h), (x+corner_size, y+h), color, 3)
        cv2.line(frame, (x+w-corner_size, y+h), (x+w, y+h), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_size), color, 3)

        # Gender label with method indicator
        label = f"[{method}] {gender.upper()}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Label background
        cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0]+10, y), color, -1)
        cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Confidence percentages
        gender_conf_text = f"Gender: {confidence*100:.1f}%"
        detection_conf_text = f"Detection: {detection_conf*100:.1f}%"

        # Position confidence texts
        conf_y = y - 5
        cv2.putText(frame, gender_conf_text, (x+w+10, conf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, detection_conf_text, (x+w+10, conf_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Analysis panel
        panel_x = x + w + 10
        panel_y = y + 40
        panel_width = 250
        panel_height = 80

        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_width, panel_y+panel_height), color, 2)

        # Panel title
        cv2.putText(frame, f"ADVANCED ANALYSIS [{method}]", (panel_x+5, panel_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Model info
        if method == "DL":
            model_info = "Deep Learning Classifier"
        else:
            model_info = "Heuristic Feature Analysis"

        cv2.putText(frame, f"Method: {model_info}", (panel_x+5, panel_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Face Conf: {detection_conf:.3f}", (panel_x+5, panel_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def draw_advanced_ui(self, frame, face_count, gender_counts, use_advanced):
        """Draw advanced UI with performance metrics"""
        # Header
        header_height = 50
        cv2.rectangle(frame, (0, 0), (640, header_height), (0, 0, 0), -1)
        cv2.putText(frame, "Advanced Gender Detection System", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Mode indicator
        mode_text = "DEEP LEARNING" if use_advanced else "HEURISTIC"
        mode_color = (0, 255, 0) if use_advanced else (255, 165, 0)
        cv2.putText(frame, f"Mode: {mode_text}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Top-right info
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()

        fps_text = f"FPS: {self.fps:.1f}"
        faces_text = f"Faces: {face_count}"

        cv2.putText(frame, fps_text, (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, faces_text, (500, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Footer
        footer_height = 30
        cv2.rectangle(frame, (0, 480-footer_height), (640, 480), (0, 0, 0), -1)

        # Instructions
        instructions = "Q:quit S:save B:benchmark M:toggle_mode H:help"
        cv2.putText(frame, instructions, (10, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Gender statistics
        male_count = gender_counts.get('male', 0)
        female_count = gender_counts.get('female', 0)
        stats_text = f"M:{male_count} F:{female_count}"
        cv2.putText(frame, stats_text, (500, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def run(self):
        """Main detection loop with advanced processing"""
        print("🚀 Starting Advanced Gender Detection...")
        print("🎯 Using SCRFD face detection + Deep Learning gender classification")
        print("⚡ Real-time processing with superior accuracy!")

        use_advanced = True  # Start with deep learning mode
        face_count = 0
        gender_counts = {'male': 0, 'female': 0, 'unknown': 0}

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break

                start_time = time.time()
                face_count, gender_counts, confidence_scores = self.process_frame(frame, use_advanced)
                processing_time = time.time() - start_time

                self.draw_advanced_ui(frame, face_count, gender_counts, use_advanced)
                cv2.imshow('Advanced Gender Detection System', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"advanced_gender_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved frame as {filename}")
                elif key == ord('b'):
                    print("🏃 Running benchmark...")
                    avg_time, fps = self.face_detector.benchmark(frame, num_runs=20)
                    print("📊 Benchmark complete!")
                elif key == ord('m'):
                    use_advanced = not use_advanced
                    mode_name = "Deep Learning" if use_advanced else "Heuristic"
                    print(f"🔄 Switched to {mode_name} mode")
                elif key == ord('h'):
                    print("💡 Help:")
                    print("  Q: Quit application")
                    print("  S: Save current frame")
                    print("  B: Run performance benchmark")
                    print("  M: Toggle between Deep Learning and Heuristic modes")
                    print("  H: Show this help")

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
    print("🧠 Advanced Gender Detection System")
    print("=" * 60)
    print("🎯 SCRFD face detection + Deep Learning gender classification")
    print("⚡ Real-time processing with superior accuracy")
    print("🎨 Toggle between DL and Heuristic modes with 'M' key")
    print("=" * 60)

    try:
        detector = LiveAdvancedGenderDetection()
        detector.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")

if __name__ == "__main__":
    main()
