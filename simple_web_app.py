#!/usr/bin/env python3
"""
Simple Web Interface for Gender Detection
Minimal backend-style interface with camera access
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import base64
import numpy as np
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
camera = None
frame_data = None
is_running = False
face_cascade = None

def initialize_face_detection():
    """Initialize simple face detection using Haar Cascade"""
    global face_cascade
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        logger.info("✅ Face detection initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize face detection: {e}")

def detect_faces_simple(image):
    """Simple face detection and gender classification"""
    results = []
    
    if face_cascade is None:
        return results
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # Simple gender classification based on face dimensions
            aspect_ratio = w / h
            face_area = w * h
            
            # Heuristic: wider faces are more likely male
            if aspect_ratio > 0.85:
                gender = 'Male'
                confidence = min(0.7, aspect_ratio)
            else:
                gender = 'Female'
                confidence = min(0.7, 1.0 - aspect_ratio)
            
            results.append({
                'bbox': (x, y, x+w, y+h),
                'gender': gender,
                'confidence': confidence
            })
    
    except Exception as e:
        logger.error(f"❌ Face detection error: {e}")
    
    return results

def draw_results(image, results):
    """Draw detection results on image"""
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        gender = result['gender']
        confidence = result['confidence']
        
        # Choose color based on gender
        color = (0, 255, 0) if gender == 'Female' else (255, 0, 0)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{gender}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image

def camera_worker():
    """Camera worker thread"""
    global camera, frame_data, is_running
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("❌ Could not open camera")
        return
    
    logger.info("📹 Camera started")
    
    while is_running:
        ret, frame = camera.read()
        if not ret:
            break
            
        # Detect faces
        results = detect_faces_simple(frame)
        
        # Draw results
        frame = draw_results(frame, results)
        
        # Encode frame as JPEG
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            frame_data = base64.b64encode(frame_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"❌ Frame encoding error: {e}")
        
        time.sleep(0.033)  # ~30 FPS
    
    camera.release()
    logger.info("📹 Camera stopped")

@app.route('/')
def index():
    """Main page"""
    return render_template('simple.html')

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    def generate():
        while is_running:
            if frame_data:
                yield f"data: {frame_data}\n\n"
            time.sleep(0.033)
    
    return Response(generate(), mimetype='text/plain')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start detection"""
    global is_running
    
    if not is_running:
        is_running = True
        camera_thread = threading.Thread(target=camera_worker, daemon=True)
        camera_thread.start()
        logger.info("🚀 Detection started")
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "already_running"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global is_running
    is_running = False
    logger.info("⏹️ Detection stopped")
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    # Initialize face detection
    initialize_face_detection()
    
    # Start Flask app
    logger.info("🚀 Starting simple web server...")
    logger.info("🌐 Access: http://172.22.196.192:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

