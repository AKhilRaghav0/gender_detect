#!/usr/bin/env python3
"""
Browser-based Gender Detection Web App
Camera access handled by browser, server processes frames
"""

from flask import Flask, render_template, Response, jsonify, request
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
face_cascade = None
detection_results = []

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
                'bbox': (int(x), int(y), int(x+w), int(y+h)),
                'gender': gender,
                'confidence': float(confidence)
            })
    
    except Exception as e:
        logger.error(f"❌ Face detection error: {e}")
    
    return results

@app.route('/')
def index():
    """Main page"""
    return render_template('browser.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame sent from browser"""
    global detection_results
    
    try:
        # Get base64 image from request
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({"error": "No image data"})
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        results = detect_faces_simple(image)
        detection_results = results
        
        return jsonify({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")
        return jsonify({"error": str(e)})

@app.route('/get_results')
def get_results():
    """Get current detection results"""
    return jsonify(detection_results)

if __name__ == '__main__':
    # Initialize face detection
    initialize_face_detection()
    
    # Start Flask app
    logger.info("🚀 Starting browser-based web server...")
    logger.info("🌐 Access: http://172.22.196.192:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
