"""
Test script for Bus Safety Gender Detection System Backend
Tests basic functionality and API endpoints
"""

import requests
import json
import base64
import time
from PIL import Image
import io
import numpy as np
import cv2 # Added missing import for cv2

# Backend configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": "/health",
    "system_info": "/system/info",
    "register_bus": "/buses/register",
    "get_bus": "/buses/{bus_id}",
    "update_status": "/buses/{bus_id}/status",
    "detect_gender": "/detect-gender",
    "gender_count": "/buses/{bus_id}/gender-count",
    "safety_metrics": "/buses/{bus_id}/safety-metrics",
    "routes": "/routes",
    "route_info": "/routes/{route_number}"
}

def create_test_image(width=640, height=480):
    """Create a simple test image for testing"""
    # Create a simple test image with some faces (simulated)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate faces
    # Face 1 (left side)
    cv2.rectangle(image, (100, 150), (200, 300), (255, 200, 150), -1)
    # Face 2 (right side)
    cv2.rectangle(image, (400, 150), (500, 300), (200, 150, 255), -1)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['health']}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['message']}")
            print(f"   Overall health: {data['data']['overall_health']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_system_info():
    """Test the system info endpoint"""
    print("\n🔍 Testing System Info...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['system_info']}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System info retrieved: {data['message']}")
            print(f"   System: {data['data']['system_name']}")
            print(f"   Version: {data['data']['version']}")
            print(f"   Routes: {', '.join(data['data']['supported_routes'])}")
            return True
        else:
            print(f"❌ System info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ System info error: {e}")
        return False

def test_bus_registration():
    """Test bus registration"""
    print("\n🔍 Testing Bus Registration...")
    try:
        bus_data = {
            "bus_id": "TEST_BUS_001",
            "route_number": "118",
            "driver_name": "Test Driver",
            "driver_contact": "+91-98765-43210",
            "capacity": 50,
            "current_location": {"lat": 28.4595, "lng": 77.0266},
            "status": "active",
            "is_active": True
        }
        
        response = requests.post(
            f"{BASE_URL}{API_ENDPOINTS['register_bus']}",
            json=bus_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Bus registered: {data['message']}")
            return "TEST_BUS_001"
        else:
            print(f"❌ Bus registration failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Bus registration error: {e}")
        return None

def test_gender_detection(bus_id):
    """Test gender detection endpoint"""
    print(f"\n🔍 Testing Gender Detection for Bus {bus_id}...")
    try:
        # Create a test image
        test_image = create_test_image()
        
        detection_data = {
            "bus_id": bus_id,
            "image_data": test_image,
            "timestamp": None
        }
        
        response = requests.post(
            f"{BASE_URL}{API_ENDPOINTS['detect_gender']}",
            json=detection_data
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Gender detection completed: {data['detected_faces']} faces detected")
            print(f"   Gender counts: {data['gender_counts']}")
            print(f"   Confidence scores: {data['confidence_scores']}")
            return True
        else:
            print(f"❌ Gender detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Gender detection error: {e}")
        return False

def test_get_bus_info(bus_id):
    """Test getting bus information"""
    print(f"\n🔍 Testing Get Bus Info for Bus {bus_id}...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['get_bus'].format(bus_id=bus_id)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Bus info retrieved: {data['bus_id']}")
            print(f"   Route: {data['route_number']}")
            print(f"   Driver: {data['driver_name']}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"❌ Get bus info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get bus info error: {e}")
        return False

def test_get_routes():
    """Test getting all routes"""
    print("\n🔍 Testing Get All Routes...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['routes']}")
        
        if response.status_code == 200:
            routes = response.json()
            print(f"✅ Routes retrieved: {len(routes)} routes found")
            for route in routes:
                print(f"   Route {route['route_number']}: {route['route_name']} (Safety: {route['safety_score']})")
            return True
        else:
            print(f"❌ Get routes failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get routes error: {e}")
        return False

def test_safety_metrics(bus_id):
    """Test getting safety metrics"""
    print(f"\n🔍 Testing Safety Metrics for Bus {bus_id}...")
    try:
        response = requests.get(f"{BASE_URL}{API_ENDPOINTS['safety_metrics'].format(bus_id=bus_id)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Safety metrics retrieved:")
            print(f"   Safety Level: {data['safety_level']}")
            print(f"   Safety Score: {data['safety_score']:.2f}")
            print(f"   Female Ratio: {data['female_ratio']:.2f}")
            print(f"   Capacity Utilization: {data['capacity_utilization']:.2f}")
            print(f"   Recommendations: {len(data['recommendations'])} items")
            return True
        else:
            print(f"❌ Safety metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Safety metrics error: {e}")
        return False

def run_all_tests():
    """Run all backend tests"""
    print("🚌 Bus Safety Gender Detection System - Backend Tests")
    print("=" * 60)
    
    # Test 1: Health Check
    if not test_health_check():
        print("❌ Backend is not running or unhealthy. Please start the backend first.")
        return False
    
    # Test 2: System Info
    if not test_system_info():
        print("❌ System info test failed.")
        return False
    
    # Test 3: Bus Registration
    bus_id = test_bus_registration()
    if not bus_id:
        print("❌ Bus registration test failed.")
        return False
    
    # Test 4: Get Bus Info
    if not test_get_bus_info(bus_id):
        print("❌ Get bus info test failed.")
        return False
    
    # Test 5: Gender Detection
    if not test_gender_detection(bus_id):
        print("❌ Gender detection test failed.")
        return False
    
    # Test 6: Get Routes
    if not test_get_routes():
        print("❌ Get routes test failed.")
        return False
    
    # Test 7: Safety Metrics
    if not test_safety_metrics(bus_id):
        print("❌ Safety metrics test failed.")
        return False
    
    print("\n🎉 All tests completed successfully!")
    print("✅ Backend is working correctly.")
    return True

if __name__ == "__main__":
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            run_all_tests()
        else:
            print("❌ Backend is not responding correctly.")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Please start the backend server first:")
        print("   cd backend")
        print("   python main.py")
    except Exception as e:
        print(f"❌ Error testing backend: {e}")
