#!/usr/bin/env python3
"""
HTTP Flask Server for Gender Detection
Fallback for testing without HTTPS
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from insightface_web_app import app

if __name__ == '__main__':
    print("🌐 Starting HTTP server (camera may not work in some browsers)...")
    print("🔒 For camera access, use HTTPS version: python3 start_https.py")
    print("🌐 Access: http://172.22.196.192:5000")
    print("📱 Mobile: http://[YOUR_IP]:5000")
    
    # Run with HTTP (camera may not work)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

