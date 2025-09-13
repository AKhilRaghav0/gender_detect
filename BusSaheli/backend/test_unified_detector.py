#!/usr/bin/env python3
"""
Test Unified Gender Detector
Verify the unified detector works correctly
"""

import sys
import os
import numpy as np
import cv2
import time

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_gender_detector import create_gender_detector, DetectionAlgorithm

def test_unified_detector():
    """Test the unified detector with different algorithms"""
    print("🧪 Testing Unified Gender Detector")
    print("=" * 50)
    
    # Create a test image (simulate a face)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face-like shape for testing
    cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)  # Face
    cv2.circle(test_image, (280, 220), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(test_image, (360, 220), 10, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(test_image, (300, 280), (340, 300), (0, 0, 0), -1)  # Nose
    cv2.rectangle(test_image, (280, 320), (360, 340), (0, 0, 0), -1)  # Mouth
    
    # Test different algorithms
    algorithms = ["haar_cascade", "professional"]
    
    for algo in algorithms:
        print(f"\n🔍 Testing {algo} algorithm...")
        
        try:
            # Create detector
            detector = create_gender_detector(algo)
            
            # Test detection
            start_time = time.time()
            results = detector.detect_gender(test_image)
            detection_time = time.time() - start_time
            
            print(f"   ✅ Detection completed in {detection_time:.3f}s")
            print(f"   📊 Found {len(results)} faces")
            
            # Test performance stats
            stats = detector.get_performance_stats()
            print(f"   📈 Performance stats: {stats}")
            
            # Test cleanup
            detector.cleanup()
            print(f"   🧹 Cleanup completed")
            
        except Exception as e:
            print(f"   ❌ Error with {algo}: {e}")
    
    print("\n🎉 Unified detector test completed!")

def test_algorithm_switching():
    """Test switching between algorithms"""
    print("\n🔄 Testing Algorithm Switching")
    print("=" * 50)
    
    detector = create_gender_detector("haar_cascade")
    print(f"Initial algorithm: {detector.algorithm.value}")
    
    # Switch to professional
    detector.switch_algorithm(DetectionAlgorithm.PROFESSIONAL)
    print(f"Switched to: {detector.algorithm.value}")
    
    # Test detection with new algorithm
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect_gender(test_image)
    print(f"Detection with {detector.algorithm.value}: {len(results)} faces")
    
    detector.cleanup()
    print("✅ Algorithm switching test completed!")

if __name__ == "__main__":
    test_unified_detector()
    test_algorithm_switching()
