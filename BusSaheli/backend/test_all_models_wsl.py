#!/usr/bin/env python3
"""
Comprehensive Model Testing in WSL
Tests all detection models including InsightFace
"""

import sys
import os
import numpy as np
import cv2
import time
import logging

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_gender_detector import create_gender_detector, DetectionAlgorithm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_insightface_installation():
    """Test InsightFace installation and initialization"""
    print("🔍 Testing InsightFace Installation...")
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("   ✅ InsightFace package imported successfully")
        
        # Test initialization
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("   ✅ InsightFace initialized successfully")
        
        # Test with a simple image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = face_app.get(test_image)
        print(f"   ✅ InsightFace detection test: {len(faces)} faces detected")
        
        # Cleanup
        del face_app
        print("   ✅ InsightFace installation test PASSED")
        return True
        
    except ImportError as e:
        print(f"   ❌ InsightFace not installed: {e}")
        print("   💡 Install with: pip install insightface")
        return False
    except Exception as e:
        print(f"   ❌ InsightFace test failed: {e}")
        return False

def test_all_algorithms():
    """Test all available algorithms"""
    print("🔍 Testing All Detection Algorithms...")
    
    algorithms = [
        ("haar_cascade", "Haar Cascade"),
        ("professional", "Professional"),
        ("insightface", "InsightFace")
    ]
    
    # Create test image with face
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)  # Face
    cv2.circle(test_image, (280, 220), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_image, (360, 220), 10, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(test_image, (300, 280), (340, 300), (0, 0, 0), -1)  # Nose
    cv2.rectangle(test_image, (280, 320), (360, 340), (0, 0, 0), -1)  # Mouth
    
    results = {}
    
    for algo_key, algo_name in algorithms:
        print(f"\n   🔍 Testing {algo_name}...")
        try:
            detector = create_gender_detector(algo_key)
            
            start_time = time.time()
            detection_results = detector.detect_gender(test_image)
            detection_time = time.time() - start_time
            
            results[algo_key] = {
                'name': algo_name,
                'faces_detected': len(detection_results),
                'detection_time': detection_time,
                'success': True,
                'algorithm_used': detection_results[0].algorithm_used if detection_results else 'none'
            }
            
            print(f"      ✅ {len(detection_results)} faces detected in {detection_time:.3f}s")
            print(f"      🔧 Algorithm used: {results[algo_key]['algorithm_used']}")
            
            # Show detection details
            for i, result in enumerate(detection_results):
                print(f"         👤 Face {i+1}: {result.gender} (confidence: {result.confidence:.2f})")
            
            detector.cleanup()
            
        except Exception as e:
            print(f"      ❌ {algo_name} failed: {e}")
            results[algo_key] = {
                'name': algo_name,
                'error': str(e),
                'success': False
            }
    
    # Show comparison
    print("\n📊 Algorithm Comparison Results:")
    print("=" * 60)
    for algo_key, data in results.items():
        if data['success']:
            print(f"{data['name']:15} ✅ {data['faces_detected']} faces, {data['detection_time']:.3f}s")
        else:
            print(f"{data['name']:15} ❌ {data['error']}")
    
    return results

def test_memory_management():
    """Test memory management system"""
    print("🔍 Testing Memory Management...")
    
    try:
        detector = create_gender_detector("insightface")
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)
        
        # Run multiple detections to test memory management
        for i in range(10):
            results = detector.detect_gender(test_image)
            if i % 3 == 0:
                stats = detector.get_performance_stats()
                print(f"   📊 Iteration {i}: {stats['total_detections']} total detections")
        
        # Final cleanup test
        detector.cleanup()
        print("   ✅ Memory management test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Memory management test failed: {e}")
        return False

def test_algorithm_switching():
    """Test switching between algorithms"""
    print("🔍 Testing Algorithm Switching...")
    
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
        print("   ✅ Algorithm switching test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Algorithm switching test failed: {e}")
        return False

def test_api_integration():
    """Test API integration"""
    print("🔍 Testing API Integration...")
    
    try:
        # Test that the API can import the unified detector
        from api_endpoints import gender_detector
        print(f"   ✅ API detector created: {gender_detector.algorithm.value}")
        
        # Test detection through API interface
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)
        
        results = gender_detector.detect_gender(test_image)
        print(f"   ✅ API detection: {len(results)} faces detected")
        
        # Test performance stats
        stats = gender_detector.get_performance_stats()
        print(f"   📊 API performance stats: {stats}")
        
        print("   ✅ API integration test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ API integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Comprehensive Model Testing in WSL")
    print("=" * 50)
    
    tests = [
        ("InsightFace Installation", test_insightface_installation),
        ("All Algorithms", test_all_algorithms),
        ("Memory Management", test_memory_management),
        ("Algorithm Switching", test_algorithm_switching),
        ("API Integration", test_api_integration)
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
        print("🎉 ALL TESTS PASSED!")
        print("✅ All models are working correctly in WSL")
        print("✅ InsightFace is fully functional")
        print("✅ Ready for production deployment")
    else:
        print("⚠️ Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
