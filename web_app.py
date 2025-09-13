#!/usr/bin/env python3
"""
Flask Web Interface for Gender Detection
Real-time camera streaming with InsightFace gender classification
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import base64
import numpy as np
from backend.insightface_gender_detection import InsightFaceGenderDetection
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
detector = None
camera = None
frame_data = None
detection_results = []
is_running = False

def initialize_detector():
    """Initialize the gender detection system"""
    global detector
    try:
        detector = InsightFaceGenderDetection()
        logger.info("✅ Gender detector initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize detector: {e}")

def camera_worker():
    """Camera worker thread"""
    global camera, frame_data, detection_results, is_running
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("❌ Could not open camera")
        return
    
    logger.info("📹 Camera started")
    
    while is_running:
        ret, frame = camera.read()
        if not ret:
            break
            
        # Process frame for detection
        if detector:
            try:
                results = detector.detect_faces_and_gender(frame)
                detection_results = results
                frame = detector.draw_results(frame, results)
            except Exception as e:
                logger.error(f"❌ Detection error: {e}")
                detection_results = []
        
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
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video feed endpoint"""
    def generate():
        while is_running:
            if frame_data:
                yield f"data: {frame_data}\n\n"
            time.sleep(0.033)
    
    return Response(generate(), mimetype='text/plain')

@app.route('/detection_data')
def detection_data():
    """Get current detection results"""
    return jsonify(detection_results)

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

@app.route('/capture_image', methods=['POST'])
def capture_image():
    """Capture current frame"""
    if frame_data:
        timestamp = int(time.time())
        filename = f"web_capture_{timestamp}.jpg"
        
        # Decode and save image
        try:
            image_data = base64.b64decode(frame_data)
            with open(filename, 'wb') as f:
                f.write(image_data)
            logger.info(f"💾 Image saved: {filename}")
            return jsonify({"status": "saved", "filename": filename})
        except Exception as e:
            logger.error(f"❌ Save error: {e}")
            return jsonify({"status": "error", "message": str(e)})
    else:
        return jsonify({"status": "no_frame"})

if __name__ == '__main__':
    # Initialize detector
    initialize_detector()
    
    # Start Flask app
    logger.info("🚀 Starting web server...")
    logger.info("🌐 Access from phone: http://YOUR_IP:5000")
    logger.info("💻 Local access: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

