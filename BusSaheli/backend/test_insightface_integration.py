#!/usr/bin/env python3
"""
Test InsightFace Integration
Verify InsightFace works correctly in the unified detector
"""

import sys
import os
import numpy as np
import cv2
import time

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_gender_detector import create_gender_detector, DetectionAlgorithm

def test_insightface_availability():
    """Test if InsightFace is available"""
    print("🔍 Testing InsightFace availability...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("   ✅ InsightFace package available")
        
        # Try to initialize
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("   ✅ InsightFace initialized successfully")
        
        # Cleanup
        del face_app
        return True
        
    except ImportError as e:
        print(f"   ❌ InsightFace not installed: {e}")
        print("   💡 Install with: pip install insightface")
        return False
    except Exception as e:
        print(f"   ❌ InsightFace initialization failed: {e}")
        return False

def test_insightface_detector():
    """Test InsightFace detector specifically"""
    print("🔍 Testing InsightFace detector...")
    
    try:
        # Create InsightFace detector
        detector = create_gender_detector("insightface")
        print(f"   ✅ Detector created: {detector.algorithm.value}")
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple face
        cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)  # Face
        cv2.circle(test_image, (280, 220), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (360, 220), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(test_image, (300, 280), (340, 300), (0, 0, 0), -1)  # Nose
        cv2.rectangle(test_image, (280, 320), (360, 340), (0, 0, 0), -1)  # Mouth
        
        # Test detection
        start_time = time.time()
        results = detector.detect_gender(test_image)
        detection_time = time.time() - start_time
        
        print(f"   ✅ Detection completed in {detection_time:.3f}s")
        print(f"   📊 Found {len(results)} faces")
        
        # Show results
        for i, result in enumerate(results):
            print(f"   👤 Face {i+1}: {result.gender} (confidence: {result.confidence:.2f})")
            print(f"      Algorithm: {result.algorithm_used}")
            print(f"      Bbox: {result.bbox}")
        
        # Test performance stats
        stats = detector.get_performance_stats()
        print(f"   📈 Performance stats: {stats}")
        
        # Test cleanup
        detector.cleanup()
        print("   🧹 Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ InsightFace detector test failed: {e}")
        return False

def test_algorithm_comparison():
    """Compare different algorithms"""
    print("🔍 Testing algorithm comparison...")
    
    algorithms = ["haar_cascade", "professional", "insightface"]
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a face
    cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)
    cv2.circle(test_image, (280, 220), 10, (0, 0, 0), -1)
    cv2.circle(test_image, (360, 220), 10, (0, 0, 0), -1)
    cv2.rectangle(test_image, (300, 280), (340, 300), (0, 0, 0), -1)
    cv2.rectangle(test_image, (280, 320), (360, 340), (0, 0, 0), -1)
    
    results_comparison = {}
    
    for algo in algorithms:
        try:
            print(f"   🔍 Testing {algo}...")
            detector = create_gender_detector(algo)
            
            start_time = time.time()
            results = detector.detect_gender(test_image)
            detection_time = time.time() - start_time
            
            results_comparison[algo] = {
                'faces_detected': len(results),
                'detection_time': detection_time,
                'algorithm_used': results[0].algorithm_used if results else 'none'
            }
            
            print(f"      ✅ {len(results)} faces in {detection_time:.3f}s")
            
            detector.cleanup()
            
        except Exception as e:
            print(f"      ❌ {algo} failed: {e}")
            results_comparison[algo] = {'error': str(e)}
    
    # Show comparison
    print("\n📊 Algorithm Comparison Results:")
    print("=" * 50)
    for algo, data in results_comparison.items():
        if 'error' in data:
            print(f"{algo:15} ❌ {data['error']}")
        else:
            print(f"{algo:15} ✅ {data['faces_detected']} faces, {data['detection_time']:.3f}s")
    
    return results_comparison

def test_algorithm_switching():
    """Test switching between algorithms"""
    print("🔍 Testing algorithm switching...")
    
    try:
        # Start with InsightFace
        detector = create_gender_detector("insightface")
        print(f"   ✅ Initial algorithm: {detector.algorithm.value}")
        
        # Switch to professional
        detector.switch_algorithm(DetectionAlgorithm.PROFESSIONAL)
        print(f"   ✅ Switched to: {detector.algorithm.value}")
        
        # Switch to Haar Cascade
        detector.switch_algorithm(DetectionAlgorithm.HAAR_CASCADE)
        print(f"   ✅ Switched to: {detector.algorithm.value}")
        
        # Switch back to InsightFace
        detector.switch_algorithm(DetectionAlgorithm.INSIGHTFACE)
        print(f"   ✅ Switched back to: {detector.algorithm.value}")
        
        detector.cleanup()
        print("   ✅ Algorithm switching test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Algorithm switching failed: {e}")
        return False

def main():
    """Run all InsightFace integration tests"""
    print("🧪 InsightFace Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("InsightFace Availability", test_insightface_availability),
        ("InsightFace Detector", test_insightface_detector),
        ("Algorithm Comparison", test_algorithm_comparison),
        ("Algorithm Switching", test_algorithm_switching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All InsightFace integration tests passed!")
        print("✅ InsightFace is ready for production use!")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
        print("💡 Make sure InsightFace is installed: pip install insightface")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
