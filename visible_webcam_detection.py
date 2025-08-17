#!/usr/bin/env python3
"""
Visible Webcam Gender Detection
This script will clearly show when the camera is working
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path

# Set environment variable for TensorFlow compatibility
os.environ['TF_USE_LEGACY_KERAS'] = 'true'

def setup_face_detector():
    """Setup face detection using OpenCV"""
    try:
        # Try to use DNN face detector first
        face_net = cv2.dnn.readNetFromTensorflow(
            'opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt'
        )
        print("✅ DNN face detector loaded")
        return face_net, 'dnn'
    except:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Haar Cascade face detector loaded")
        return face_cascade, 'haar'

def detect_faces_haar(face_cascade, frame):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    confidences = [1.0] * len(faces)  # Haar doesn't provide confidence
    
    # Convert numpy array to list format
    faces_list = []
    for (x, y, w, h) in faces:
        faces_list.append([x, y, w, h])
    return faces_list, confidences

def simple_gender_prediction(face_img):
    """
    Simple gender prediction based on face features
    This is a placeholder - you can replace with your trained model
    """
    # For now, return a random prediction
    # In production, you would load your trained model here
    import random
    gender = random.choice(['man', 'woman'])
    confidence = random.uniform(0.6, 0.95)
    
    return gender, confidence

def process_frame(frame, face_detector, detector_type):
    """Process a single frame for face and gender detection"""
    
    # Detect faces
    if detector_type == 'dnn':
        # DNN detection code would go here
        faces, face_confidences = [], []
    else:
        faces, face_confidences = detect_faces_haar(face_detector, frame)
    
    results = []
    
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size > 0:
            # Predict gender (placeholder for now)
            gender, gender_confidence = simple_gender_prediction(face_img)
            
            results.append({
                'bbox': (x, y, w, h),
                'gender': gender,
                'gender_confidence': gender_confidence,
                'face_confidence': face_confidences[i]
            })
    
    return results

def draw_results(frame, results):
    """Draw detection results on frame"""
    for result in results:
        x, y, w, h = result['bbox']
        gender = result['gender']
        gender_conf = result['gender_confidence']
        
        # Choose color based on gender
        color = (255, 0, 0) if gender == 'man' else (0, 255, 255)  # Blue for man, Yellow for woman
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label
        label = f"{gender}: {gender_conf:.2f}"
        
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
    
    return frame

def run_webcam(camera_id=0):
    """Run real-time face and gender detection on webcam"""
    print(f"🎥 Starting webcam detection (Camera ID: {camera_id})")
    print("Press 'q' to quit, 's' to save screenshot")
    print("Note: Gender prediction is currently using placeholder logic")
    print("To use your trained model, replace simple_gender_prediction() function")
    
    # Setup face detector
    face_detector, detector_type = setup_face_detector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully!")
    print("🖼️  Camera window should appear now...")
    print("If you don't see it, check:")
    print("1. Look for a window titled 'Visible Gender Detection'")
    print("2. Check if it's minimized in the taskbar")
    print("3. Look behind other windows")
    
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            results = process_frame(frame, face_detector, detector_type)
            
            # Draw results
            frame_with_results = draw_results(frame, results)
            
            # Add status text to make it obvious the camera is working
            status_text = f"Camera Active - Press 'q' to quit"
            cv2.putText(
                frame_with_results,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Calculate and display FPS
            fps_counter += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = fps_counter / elapsed_time
                fps_counter = 0
                start_time = time.time()
                
                # Display FPS on frame
                cv2.putText(
                    frame_with_results,
                    f"FPS: {fps:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            # Display frame
            window_name = 'Visible Gender Detection - CAMERA IS WORKING!'
            cv2.imshow(window_name, frame_with_results)
            
            # Make sure the window is visible
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quit key pressed")
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_results)
                print(f"📸 Screenshot saved: {filename}")
    
    except Exception as e:
        print(f"❌ Error during webcam operation: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Webcam session ended")

def main():
    """Main function"""
    print("=" * 70)
    print("🎥 VISIBLE Webcam Gender Detection")
    print("=" * 70)
    print("This script will clearly show when the camera is working!")
    print("Look for a window titled 'Visible Gender Detection - CAMERA IS WORKING!'")
    print()
    
    try:
        run_webcam()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
