#!/usr/bin/env python3
"""
Download SCRFD model for gender detection system
"""

import os
import sys

def main():
    print("🤖 SCRFD Model Downloader")
    print("=" * 40)
    print("📋 Since automatic download failed, please download manually:")
    print()
    print("🔗 BROWSER DOWNLOAD:")
    print("1. Open this URL in your browser:")
    print("   https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g.onnx")
    print()
    print("2. Save the file as: scrfd_2.5g.onnx")
    print()
    print("3. Place it in this folder:")
    print("   gender_detect/backend/models/scrfd_2.5g.onnx")
    print()
    print("📂 Creating models directory...")

    # Create models directory
    models_dir = "backend/models"
    os.makedirs(models_dir, exist_ok=True)
    print(f"✅ Created directory: {models_dir}")

    expected_path = os.path.join(models_dir, "scrfd_2.5g.onnx")
    print(f"📍 Expected model location: {expected_path}")
    print()
    print("🎯 ALTERNATIVE: Try this direct download command:")
    print("powershell -Command \"Invoke-WebRequest -Uri 'https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g.onnx' -OutFile 'backend/models/scrfd_2.5g.onnx'\"")
    print()
    print("🚀 After downloading, run:")
    print("python backend/live_advanced_gender_detection.py")

if __name__ == "__main__":
    main()