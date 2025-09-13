#!/usr/bin/env python3
"""
Camera Setup Test for Gender Detection System
Tests camera access in Windows, WSL, and various configurations
"""

import cv2
import platform
import sys
import os

def detect_environment():
    """Detect current environment (Windows/WSL/Linux/macOS)"""
    system = platform.system().lower()

    # Check for WSL
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            is_wsl = 'microsoft' in version or 'wsl' in version
    except:
        is_wsl = False

    if is_wsl:
        return "WSL (Linux on Windows)"
    elif system == "windows":
        return "Windows"
    elif system == "linux":
        return "Linux"
    elif system == "darwin":
        return "macOS"
    else:
        return f"Unknown ({system})"

def test_opencv():
    """Test OpenCV installation and camera support"""
    print("🔍 Testing OpenCV...")
    try:
        print(f"   OpenCV version: {cv2.__version__}")
        print("   ✅ OpenCV working")
        return True
    except Exception as e:
        print(f"   ❌ OpenCV error: {e}")
        return False

def test_camera_access():
    """Test camera access with different methods"""
    print("\n📹 Testing Camera Access...")

    results = {}

    # Test direct camera indices
    print("   Testing camera indices (0-4)...")
    working_cameras = []

    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working_cameras.append({
                        'index': i,
                        'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                        'channels': frame.shape[2] if len(frame.shape) > 2 else 1
                    })
                    print(f"   ✅ Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    print(f"   ⚠️ Camera {i}: Opened but no frame")
            else:
                print(f"   ❌ Camera {i}: Not accessible")
            cap.release()
        except Exception as e:
            print(f"   ❌ Camera {i}: Error - {e}")

    results['direct_cameras'] = working_cameras

    # Test IP camera (common phone IP webcam)
    print("\n   Testing IP camera access...")
    ip_cameras = [
        "http://192.168.1.100:8080/video",
        "http://10.0.0.100:8080/video",
        "rtsp://192.168.1.100:554/live"
    ]

    for ip_url in ip_cameras:
        try:
            cap = cv2.VideoCapture(ip_url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"   ✅ IP Camera: {ip_url} - {frame.shape[1]}x{frame.shape[0]}")
                    results['ip_camera'] = ip_url
                    break
            cap.release()
        except:
            continue

    if 'ip_camera' not in results:
        print("   ⚠️ No IP cameras accessible")
        results['ip_camera'] = None

    return results

def test_gender_detection_pipeline():
    """Test the complete gender detection pipeline"""
    print("\n🎯 Testing Gender Detection Pipeline...")

    try:
        # Test SCRFD detector
        sys.path.append('backend')
        from backend.scrfd_detection import create_scrfd_detector

        detector = create_scrfd_detector()
        print("   ✅ SCRFD detector created")

        # Test with dummy image
        dummy_img = cv2.imread('test.jpg') if os.path.exists('test.jpg') else None
        if dummy_img is None:
            # Create dummy image
            dummy_img = cv2.imread('backend/test.jpg') if os.path.exists('backend/test.jpg') else None

        if dummy_img is not None:
            faces = detector.detect_faces(dummy_img)
            print(f"   ✅ Face detection working: {len(faces)} faces detected")
        else:
            print("   ⚠️ No test image found - face detection not tested")

        return True

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Pipeline error: {e}")
        return False

def show_recommendations(environment, camera_results):
    """Show personalized recommendations"""
    print("\n🎯 RECOMMENDATIONS:")

    if environment == "WSL (Linux on Windows)":
        print("🐧 WSL DETECTED:")
        if camera_results['direct_cameras']:
            print("   ✅ Direct camera access working!")
            print("   🎉 You can use WSL for camera (rare case)")
        else:
            print("   ⚠️ Limited camera support in WSL")
            print("   💡 RECOMMENDED: Use Windows Python instead")
            print("   📋 Run: python backend/live_scrfd_detection.py (from Windows)")

        if camera_results['ip_camera']:
            print("   📱 IP camera working - great for WSL!")

    elif environment == "Windows":
        print("🪟 WINDOWS DETECTED:")
        if camera_results['direct_cameras']:
            print("   ✅ Camera access perfect!")
            print("   🚀 Ready to run gender detection")
        else:
            print("   ⚠️ Camera not accessible")
            print("   💡 Check Windows Camera app and permissions")

    # General recommendations
    print("\n📋 GENERAL TIPS:")
    print("   • Test: python backend/live_scrfd_detection.py")
    print("   • Debug: python test_camera_setup.py")
    print("   • Help: python backend/live_scrfd_detection.py --help")

def main():
    """Main test function"""
    print("🎥 Camera Setup Test - Advanced Gender Detection System")
    print("=" * 60)

    # Detect environment
    environment = detect_environment()
    print(f"🔍 Environment: {environment}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()

    # Test OpenCV
    opencv_ok = test_opencv()

    if not opencv_ok:
        print("❌ OpenCV not working - install: pip install opencv-python")
        return

    # Test camera access
    camera_results = test_camera_access()

    # Test pipeline
    pipeline_ok = test_gender_detection_pipeline()

    # Show summary
    print("\n📊 SUMMARY:")
    print(f"   Environment: {environment}")
    print(f"   OpenCV: {'✅' if opencv_ok else '❌'}")
    print(f"   Direct Cameras: {len(camera_results['direct_cameras'])} found")
    print(f"   IP Camera: {'✅' if camera_results['ip_camera'] else '❌'}")
    print(f"   Pipeline: {'✅' if pipeline_ok else '❌'}")

    # Show recommendations
    show_recommendations(environment, camera_results)

    print("\n" + "=" * 60)
    if camera_results['direct_cameras'] or camera_results['ip_camera']:
        print("🎉 Camera setup successful! Ready for gender detection!")
    else:
        print("⚠️ Camera setup needed. Check recommendations above.")

if __name__ == "__main__":
    main()


