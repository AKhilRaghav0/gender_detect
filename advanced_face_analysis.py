#!/usr/bin/env python3
"""
Advanced Face Analysis System
Using DeepFace library for comprehensive face analysis
"""

import cv2
import numpy as np
import time
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append('backend')

try:
    from deepface import DeepFace
    print("✅ DeepFace imported successfully!")
except ImportError:
    print("❌ DeepFace not installed. Installing...")
    os.system("pip install deepface")
    try:
        from deepface import DeepFace
        print("✅ DeepFace installed and imported!")
    except ImportError:
        print("❌ Failed to install DeepFace")
        sys.exit(1)

class AdvancedFaceAnalyzer:
    def __init__(self):
        """Initialize advanced face analyzer"""
        self.analysis_models = {
            'gender': 'gender',
            'age': 'age',
            'emotion': 'emotion',
            'race': 'race'
        }

        print("🎭 Advanced Face Analyzer initialized!")
        print("🧠 Using DeepFace for comprehensive analysis")

    def analyze_face(self, face_image):
        """
        Analyze single face image

        Args:
            face_image: Face ROI (BGR format)

        Returns:
            dict: Analysis results
        """
        try:
            # Convert BGR to RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Analyze with DeepFace
            results = DeepFace.analyze(face_rgb,
                                     actions=['gender', 'age', 'emotion', 'race'],
                                     enforce_detection=False,
                                     silent=True)

            # Extract results (DeepFace returns list for single image)
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            return {
                'gender': result.get('dominant_gender', 'Unknown'),
                'gender_confidence': result.get('gender', {}),
                'age': result.get('age', 'Unknown'),
                'emotion': result.get('dominant_emotion', 'Unknown'),
                'emotion_confidence': result.get('emotion', {}),
                'race': result.get('dominant_race', 'Unknown'),
                'race_confidence': result.get('race', {}),
                'success': True
            }

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {
                'gender': 'Unknown',
                'age': 'Unknown',
                'emotion': 'Unknown',
                'race': 'Unknown',
                'success': False,
                'error': str(e)
            }

    def analyze_frame(self, frame):
        """
        Analyze all faces in frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            list: List of face analysis results
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect and analyze faces
            results = DeepFace.analyze(frame_rgb,
                                     actions=['gender', 'age', 'emotion', 'race'],
                                     enforce_detection=False,
                                     silent=True)

            # Ensure results is a list
            if not isinstance(results, list):
                results = [results]

            # Process results
            analyses = []
            for result in results:
                # Get face region (approximate)
                face_region = self._extract_face_region(frame, result)

                analysis = {
                    'bbox': face_region.get('bbox', (0, 0, 100, 100)),
                    'gender': result.get('dominant_gender', 'Unknown'),
                    'age': result.get('age', 'Unknown'),
                    'emotion': result.get('dominant_emotion', 'Unknown'),
                    'race': result.get('dominant_race', 'Unknown'),
                    'confidence': {
                        'gender': result.get('gender', {}),
                        'emotion': result.get('emotion', {}),
                        'race': result.get('race', {})
                    }
                }
                analyses.append(analysis)

            return analyses

        except Exception as e:
            print(f"❌ Frame analysis error: {e}")
            return []

    def _extract_face_region(self, frame, analysis_result):
        """Extract face region from DeepFace result"""
        # DeepFace doesn't always return face coordinates
        # This is a fallback to estimate face location
        h, w = frame.shape[:2]

        # Default face region (center of frame)
        face_size = min(w, h) // 4
        center_x, center_y = w // 2, h // 2

        return {
            'bbox': (center_x - face_size//2, center_y - face_size//2,
                    face_size, face_size)
        }

class LiveAdvancedFaceAnalysis:
    def __init__(self):
        """Initialize live advanced face analysis"""
        self.face_analyzer = AdvancedFaceAnalyzer()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open webcam!")

        # Performance tracking
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0.0

        print("🎥 Webcam initialized successfully!")
        print("🧠 DeepFace analysis ready!")
        print("💡 Press 'q' to quit, 's' to save frame, 'a' to analyze single frame")

    def process_frame(self, frame, analyze_faces=True):
        """Process frame for advanced face analysis"""
        if not analyze_faces:
            return [], frame

        # Analyze faces in frame
        face_analyses = self.face_analyzer.analyze_frame(frame)

        # Draw results
        for analysis in face_analyses:
            self._draw_analysis_results(frame, analysis)

        return face_analyses, frame

    def _draw_analysis_results(self, frame, analysis):
        """Draw analysis results on frame"""
        bbox = analysis['bbox']
        x, y, w, h = bbox

        # Choose color based on gender
        if analysis['gender'].lower() == 'man':
            color = (0, 165, 255)  # Orange for male
        elif analysis['gender'].lower() == 'woman':
            color = (255, 0, 255)  # Magenta for female
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw corner indicators
        corner_size = 8
        cv2.line(frame, (x, y), (x+corner_size, y), color, 3)
        cv2.line(frame, (x, y), (x, y+corner_size), color, 3)
        cv2.line(frame, (x+w-corner_size, y), (x+w, y), color, 3)
        cv2.line(frame, (x+w, y), (x+w, y+corner_size), color, 3)
        cv2.line(frame, (x, y+h-corner_size), (x, y+h), color, 3)
        cv2.line(frame, (x, y+h), (x+corner_size, y+h), color, 3)
        cv2.line(frame, (x+w-corner_size, y+h), (x+w, y+h), color, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_size), color, 3)

        # Analysis panel
        panel_x = x + w + 10
        panel_y = y
        panel_width = 280
        panel_height = 140

        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x+panel_width, panel_y+panel_height),
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y),
                     (panel_x+panel_width, panel_y+panel_height),
                     color, 2)

        # Panel title
        cv2.putText(frame, "DEEPFACE ANALYSIS", (panel_x+5, panel_y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Results
        y_offset = 45
        cv2.putText(frame, f"Gender: {analysis['gender']}", (panel_x+5, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20
        cv2.putText(frame, f"Age: {analysis['age']}", (panel_x+5, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20
        cv2.putText(frame, f"Emotion: {analysis['emotion']}", (panel_x+5, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 20
        cv2.putText(frame, f"Ethnicity: {analysis['race']}", (panel_x+5, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Confidence info
        y_offset += 25
        cv2.putText(frame, "CONFIDENCE SCORES:", (panel_x+5, panel_y+y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_ui(self, frame, face_count, analysis_enabled):
        """Draw user interface"""
        # Header
        header_height = 60
        cv2.rectangle(frame, (0, 0), (640, header_height), (0, 0, 0), -1)
        cv2.putText(frame, "Advanced Face Analysis System", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Powered by DeepFace", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Mode indicator
        mode_text = "ANALYSIS: ON" if analysis_enabled else "ANALYSIS: OFF"
        mode_color = (0, 255, 0) if analysis_enabled else (0, 0, 255)
        cv2.putText(frame, mode_text, (450, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # Performance metrics
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()

        fps_text = f"FPS: {self.fps:.1f}"
        faces_text = f"Faces: {face_count}"
        cv2.putText(frame, fps_text, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, faces_text, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Footer
        footer_height = 35
        cv2.rectangle(frame, (0, 480-footer_height), (640, 480), (0, 0, 0), -1)
        instructions = "Q:quit S:save A:toggle_analysis H:help"
        cv2.putText(frame, instructions, (10, 475),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self):
        """Main analysis loop"""
        print("🚀 Starting Advanced Face Analysis...")
        print("🧠 Using DeepFace for comprehensive face analysis")
        print("✨ Features: Gender, Age, Emotion, Ethnicity detection")

        analysis_enabled = True
        face_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break

                # Process frame
                start_time = time.time()
                face_analyses, frame = self.process_frame(frame, analysis_enabled)
                face_count = len(face_analyses)
                processing_time = time.time() - start_time

                # Draw UI
                self.draw_ui(frame, face_count, analysis_enabled)

                # Display frame
                cv2.imshow('Advanced Face Analysis - DeepFace', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"deepface_analysis_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Saved frame as {filename}")
                elif key == ord('a'):
                    analysis_enabled = not analysis_enabled
                    status = "ENABLED" if analysis_enabled else "DISABLED"
                    print(f"🔄 Face analysis {status}")
                elif key == ord('h'):
                    print("\n💡 Help:")
                    print("  Q: Quit application")
                    print("  S: Save current frame")
                    print("  A: Toggle face analysis on/off")
                    print("  H: Show this help")
                    print("\n🎯 Analysis includes:")
                    print("  • Gender detection (Male/Female)")
                    print("  • Age estimation")
                    print("  • Emotion recognition")
                    print("  • Ethnicity classification")
                    print()

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
    print("🧠 Advanced Face Analysis System")
    print("=" * 50)
    print("✨ Using DeepFace for comprehensive analysis")
    print("🎯 Features: Gender, Age, Emotion, Ethnicity")
    print("⚡ Real-time processing with advanced AI")
    print("=" * 50)

    try:
        analyzer = LiveAdvancedFaceAnalysis()
        analyzer.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("\n💡 Make sure your webcam is available and DeepFace is installed:")
        print("   pip install deepface")

if __name__ == "__main__":
    main()
