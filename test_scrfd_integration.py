#!/usr/bin/env python3
"""
Test script for SCRFD Integration in Gender Detection System
"""

import sys
import os
sys.path.append('backend')

from backend.scrfd_detection import create_scrfd_detector
import cv2
import numpy as np

def test_scrfd_basic():
    """Test basic SCRFD functionality"""
    print("🧪 Testing SCRFD Face Detection...")

    # Create detector
    detector = create_scrfd_detector(conf_threshold=0.5)
    print("✅ SCRFD detector created successfully!")

    # Test with a simple image (create a dummy image if no webcam)
    try:
        # Try to capture from webcam
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Test face detection
                faces = detector.detect_faces(frame)
                print(f"✅ Detected {len(faces)} faces in webcam frame")

                # Show results
                for i, (x, y, w, h, conf) in enumerate(faces):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, ".2f", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("SCRFD Test Results", frame)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()

            cap.release()
        else:
            print("⚠️ Webcam not available - creating test pattern")

            # Create a test image
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_img, "SCRFD Integration Test", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            faces = detector.detect_faces(test_img)
            print(f"📊 Test completed - detector ready (found {len(faces)} faces in test image)")

    except Exception as e:
        print(f"❌ Error during testing: {e}")

def test_live_system():
    """Test the complete live SCRFD system"""
    print("\n🎥 Testing Live SCRFD Gender Detection System...")

    try:
        from backend.live_scrfd_detection import LiveSCRFDetection

        print("✅ Live SCRFD Detection system imported successfully!")
        print("📋 System Features:")
        print("   • SCRFD face detection with Haar cascade fallback")
        print("   • Real-time gender analysis")
        print("   • Polished UI with analysis panels")
        print("   • Confidence scoring and debugging")
        print("   • Benchmarking capabilities")

        # Create instance to test initialization
        detector = LiveSCRFDetection()
        print("✅ Live SCRFD Detection system initialized!")

        print("\n🚀 Ready to run! Use this command:")
        print("   python backend/live_scrfd_detection.py")

    except Exception as e:
        print(f"❌ Error initializing live system: {e}")

def main():
    """Main test function"""
    print("🤖 SCRFD Integration Test Suite")
    print("=" * 50)

    # Test basic SCRFD functionality
    test_scrfd_basic()

    # Test live system
    test_live_system()

    print("\n" + "=" * 50)
    print("✅ SCRFD Integration Test Complete!")
    print("\n📋 Next Steps:")
    print("1. Download SCRFD model manually for full performance:")
    print("   https://github.com/deepinsight/insightface/releases")
    print("2. Place scrfd_2.5g.onnx in backend/models/")
    print("3. Run: python backend/live_scrfd_detection.py")
    print("\n🎯 The system works with Haar cascade fallback until SCRFD model is available!")

if __name__ == "__main__":
    main()
