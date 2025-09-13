#!/usr/bin/env python3
"""
Test API Endpoints
Verify all API endpoints work correctly
"""

import requests
import json
import base64
import numpy as np
import cv2
import time

# API base URL
BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a test image for API testing"""
    # Create a simple test image with a face-like shape
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a face
    cv2.rectangle(image, (200, 150), (440, 350), (255, 255, 255), -1)  # Face
    cv2.circle(image, (280, 220), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(image, (360, 220), 10, (0, 0, 0), -1)  # Right eye
    cv2.rectangle(image, (300, 280), (340, 300), (0, 0, 0), -1)  # Nose
    cv2.rectangle(image, (280, 320), (360, 340), (0, 0, 0), -1)  # Mouth
    
    # Encode as base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{image_base64}"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed: {data['status']}")
            print(f"   📊 Redis connected: {data['redis_connected']}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_gender_detection():
    """Test gender detection endpoint"""
    print("🔍 Testing gender detection endpoint...")
    
    try:
        test_image = create_test_image()
        
        payload = {
            "bus_id": "test_bus_001",
            "image_data": test_image,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/detect-gender", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Gender detection successful")
            print(f"   📊 Passengers: {data['passenger_count']}")
            print(f"   👥 Female: {data['female_count']}, Male: {data['male_count']}")
            print(f"   🛡️ Safety Score: {data['safety_score']:.2f}")
            print(f"   ⏱️ Processing Time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"   ❌ Gender detection failed: {response.status_code}")
            print(f"   📝 Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Gender detection error: {e}")
        return False

def test_detector_status():
    """Test detector status endpoint"""
    print("🔍 Testing detector status endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/detector/status")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Detector status retrieved")
            print(f"   🔧 Algorithm: {data['algorithm']}")
            print(f"   📊 Status: {data['status']}")
            return True
        else:
            print(f"   ❌ Detector status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Detector status error: {e}")
        return False

def test_algorithm_switch():
    """Test algorithm switching endpoint"""
    print("🔍 Testing algorithm switch endpoint...")
    
    try:
        # Switch to haar_cascade
        response = requests.post(f"{BASE_URL}/api/v1/detector/switch", json="haar_cascade")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Algorithm switch successful")
            print(f"   🔧 Current algorithm: {data['current_algorithm']}")
            return True
        else:
            print(f"   ❌ Algorithm switch failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Algorithm switch error: {e}")
        return False

def test_routes_endpoint():
    """Test routes endpoint"""
    print("🔍 Testing routes endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/routes")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Routes retrieved: {len(data)} routes")
            return True
        else:
            print(f"   ❌ Routes failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Routes error: {e}")
        return False

def test_statistics():
    """Test statistics endpoint"""
    print("🔍 Testing statistics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/statistics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Statistics retrieved")
            print(f"   📊 Active routes: {data['active_routes']}")
            return True
        else:
            print(f"   ❌ Statistics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Statistics error: {e}")
        return False

def test_safety_alert():
    """Test safety alert creation"""
    print("🔍 Testing safety alert creation...")
    
    try:
        payload = {
            "bus_id": "test_bus_001",
            "alert_type": "crowd_density",
            "message": "High crowd density detected",
            "severity": "medium"
        }
        
        response = requests.post(f"{BASE_URL}/api/v1/alerts", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Safety alert created: {data['message']}")
            return True
        else:
            print(f"   ❌ Safety alert failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Safety alert error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 API Endpoints Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Gender Detection", test_gender_detection),
        ("Detector Status", test_detector_status),
        ("Algorithm Switch", test_algorithm_switch),
        ("Routes", test_routes_endpoint),
        ("Statistics", test_statistics),
        ("Safety Alert", test_safety_alert)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the API server and logs.")
    
    return passed == total

if __name__ == "__main__":
    print("⚠️ Make sure the API server is running on http://localhost:8000")
    print("   Start it with: python api_endpoints.py")
    print()
    
    success = main()
    exit(0 if success else 1)
