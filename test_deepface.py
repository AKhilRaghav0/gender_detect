#!/usr/bin/env python3
"""
Test DeepFace functionality
"""

import cv2
import numpy as np
from deepface import DeepFace

def test_deepface():
    """Test DeepFace basic functionality"""
    print("🧠 Testing DeepFace...")

    # Create a simple test image
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    try:
        # Test basic analysis
        result = DeepFace.analyze(test_img,
                                actions=['gender', 'age', 'emotion'],
                                enforce_detection=False,
                                silent=True)

        print("✅ DeepFace analysis successful!")
        print(f"📊 Result: {result[0] if isinstance(result, list) else result}")

        # Test webcam
        print("📹 Testing webcam...")
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Resize for faster processing
                frame_small = cv2.resize(frame, (320, 240))

                # Analyze frame
                results = DeepFace.analyze(frame_small,
                                         actions=['gender', 'age', 'emotion'],
                                         enforce_detection=False,
                                         silent=True)

                print(f"✅ Webcam analysis successful! Found {len(results) if isinstance(results, list) else 1} face(s)")

                # Show result
                result = results[0] if isinstance(results, list) else results
                print(f"🎯 Gender: {result.get('dominant_gender', 'Unknown')}")
                print(f"🎂 Age: {result.get('age', 'Unknown')}")
                print(f"😊 Emotion: {result.get('dominant_emotion', 'Unknown')}")

            cap.release()
        else:
            print("⚠️ Webcam not available")

        return True

    except Exception as e:
        print(f"❌ DeepFace test failed: {e}")
        return False

if __name__ == "__main__":
    print("🤖 DeepFace Test")
    print("=" * 30)

    success = test_deepface()

    if success:
        print("\n🎉 DeepFace is working perfectly!")
        print("🚀 You can now run advanced face analysis!")
        print("💡 Try: python -c \"from deepface import DeepFace; print('DeepFace ready!')\"")
    else:
        print("\n⚠️ DeepFace test failed")
