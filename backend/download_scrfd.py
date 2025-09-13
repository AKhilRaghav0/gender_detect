#!/usr/bin/env python3
"""
Download SCRFD model for gender detection system
"""

import os
import sys

def main():
    print("🤖 SCRFD Model Downloader")
    print("=" * 40)
    print("📋 Manual download instructions (automatic download failed):")
    print()
    print("🔗 BROWSER DOWNLOAD:")
    print("1. Click this link or copy to browser:")
    print("   https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g.onnx")
    print()
    print("2. Save the file as: scrfd_2.5g.onnx")
    print()
    print("3. Place it in this folder:")
    current_dir = os.getcwd()
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"   {models_dir}\\scrfd_2.5g.onnx")
    print()
    print("📂 Created models directory:", models_dir)
    print()
    print("🎯 ALTERNATIVE: PowerShell download command:")
    print("Invoke-WebRequest -Uri 'https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g.onnx' -OutFile 'models\\scrfd_2.5g.onnx'")
    print()
    print("🚀 After downloading, run:")
    print("python live_advanced_gender_detection.py")
    print()
    print("💡 The system will work with Haar cascades until SCRFD model is downloaded!")

if __name__ == "__main__":
    main()
