#!/usr/bin/env python3
"""
HTTPS Flask Server for Gender Detection
Required for camera access in modern browsers
"""

from flask import Flask, render_template, request, jsonify
import ssl
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from insightface_web_app import app

if __name__ == '__main__':
    # Create SSL context for HTTPS
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    
    # Generate self-signed certificate (for development only)
    # In production, use a real certificate
    try:
        context.load_cert_chain('cert.pem', 'key.pem')
        print("✅ Using existing SSL certificate")
    except FileNotFoundError:
        print("⚠️  No SSL certificate found, generating self-signed certificate...")
        print("   This will show a security warning in browsers - click 'Advanced' and 'Proceed'")
        
        # Generate self-signed certificate
        import subprocess
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096', '-keyout', 'key.pem', 
            '-out', 'cert.pem', '-days', '365', '-nodes', '-subj', 
            '/C=US/ST=State/L=City/O=Organization/CN=localhost'
        ], check=True)
        
        context.load_cert_chain('cert.pem', 'key.pem')
        print("✅ Self-signed certificate generated")
    
    print("🔒 Starting HTTPS server...")
    print("🌐 Access: https://172.22.196.192:5000")
    print("📱 Mobile: https://[YOUR_IP]:5000")
    print("⚠️  Accept security warning for self-signed certificate")
    
    # Run with HTTPS
    app.run(host='0.0.0.0', port=5000, ssl_context=context, debug=False, threaded=True)

