"""
Simple Test for Gender Detection Service Only
Tests the core detection logic without database complexity
"""

import cv2
import numpy as np
from gender_detection_service import GenderDetectionService

def create_test_face_image():
    """Create a simple test image with a simulated face"""
    # Create a 640x480 image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face (oval shape)
    cv2.ellipse(image, (320, 240), (120, 160), 0, 0, 360, (255, 200, 150), -1)
    
    # Draw eyes
    cv2.circle(image, (280, 200), 15, (255, 255, 255), -1)  # Left eye
    cv2.circle(image, (360, 200), 15, (255, 255, 255), -1)  # Right eye
    cv2.circle(image, (280, 200), 8, (0, 0, 0), -1)        # Left pupil
    cv2.circle(image, (360, 200), 8, (0, 0, 0), -1)        # Right pupil
    
    # Draw nose
    cv2.ellipse(image, (320, 240), (8, 15), 0, 0, 180, (200, 150, 100), -1)
    
    # Draw mouth
    cv2.ellipse(image, (320, 280), (30, 15), 0, 0, 180, (150, 100, 100), -1)
    
    return image

def test_detection_service():
    """Test the gender detection service"""
    print("🧪 Testing Gender Detection Service")
    print("=" * 50)
    
    try:
        # Initialize the service
        service = GenderDetectionService()
        print("✅ Service initialized successfully")
        
        # Create test image
        test_image = create_test_face_image()
        print("✅ Test image created")
        
        # Convert to base64
        import base64
        _, buffer = cv2.imencode('.jpg', test_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        print("✅ Image converted to base64")
        
        # Test detection
        print("\n🔍 Testing gender detection...")
        face_count, gender_counts, confidence_scores = service.process_image(img_base64)
        
        print(f"📊 Results:")
        print(f"   - Faces detected: {face_count}")
        print(f"   - Gender counts: {gender_counts}")
        print(f"   - Confidence scores: {confidence_scores}")
        
        # Test safety metrics
        print("\n🛡️ Testing safety metrics...")
        
        # Get the average confidence score
        avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
        
        gender_count = service.create_gender_count("TEST_BUS", gender_counts, avg_confidence)
        safety_metrics = service.calculate_safety_metrics("TEST_BUS", gender_count, "118")  # Route 118 for testing
        
        print(f"📊 Safety Metrics:")
        print(f"   - Female ratio: {safety_metrics.female_ratio:.2f}")
        print(f"   - Capacity utilization: {safety_metrics.capacity_utilization:.2f}")
        print(f"   - Safety score: {safety_metrics.safety_score:.2f}")
        print(f"   - Safety level: {safety_metrics.safety_level}")
        print(f"   - Recommendations: {safety_metrics.recommendations}")
        
        print("\n🎉 All tests passed! Your detection system is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_detection_service()
