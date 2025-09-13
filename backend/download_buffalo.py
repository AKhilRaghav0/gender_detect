#!/usr/bin/env python3
"""
Download and extract Buffalo SCRFD model
"""

import os
import zipfile
import urllib.request
import sys

def download_buffalo_model():
    """Download and extract buffalo_s model containing SCRFD"""
    print("⬇️ Downloading Buffalo SCRFD model...")

    models_dir = "models"
    zip_path = os.path.join(models_dir, "buffalo_s.zip")
    extract_path = os.path.join(models_dir, "buffalo_s")

    # Download URL
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"

    try:
        print(f"📡 Downloading from: {url}")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download completed!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

    # Extract the zip file
    try:
        print("📦 Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction completed!")

        # List extracted files
        print("\n📁 Extracted files:")
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('.onnx'):
                    print(f"  🎯 {os.path.join(root, file)}")

        # Look for SCRFD model
        scrfd_path = None
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if 'scrfd' in file.lower() and file.endswith('.onnx'):
                    scrfd_path = os.path.join(root, file)
                    break

        if scrfd_path:
            # Copy to expected location
            expected_path = os.path.join(models_dir, "scrfd_2.5g.onnx")
            import shutil
            shutil.copy2(scrfd_path, expected_path)
            print(f"✅ SCRFD model ready: {expected_path}")
            return True
        else:
            print("⚠️ SCRFD model not found in extracted files")
            return False

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

if __name__ == "__main__":
    print("🦬 Buffalo SCRFD Model Downloader")
    print("=" * 40)

    success = download_buffalo_model()

    if success:
        print("\n🎉 Buffalo model downloaded and SCRFD extracted!")
        print("🚀 You can now run: python live_advanced_gender_detection.py")
    else:
        print("\n⚠️ Download/extraction failed. Try manual download:")
        print("1. Download: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip")
        print("2. Extract to: backend/models/buffalo_s/")
        print("3. Look for scrfd_*.onnx file and copy to backend/models/scrfd_2.5g.onnx")
