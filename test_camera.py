#!/usr/bin/env python3
"""
Simple camera test script to debug webcam access
"""

import cv2
import numpy as np
import time

def test_camera():
    """Test basic camera functionality"""
    
    print("🎥 Testing camera access...")
    
    # Try to open camera 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera 0")
        return False
    
    print("✅ Camera 0 opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Camera resolution: {width}x{height}")
    print(f"🎬 Camera FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame from camera")
        cap.release()
        return False
    
    print(f"✅ Frame read successfully: {frame.shape}")
    
    # Display the frame
    print("🖼️  Displaying camera frame...")
    print("Press any key to continue...")
    
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Test continuous capture
    print("🎥 Testing continuous capture (5 seconds)...")
    print("Press 'q' to quit early...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Camera Test - Continuous', frame)
            
            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # Small delay to make it visible
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"✅ Continuous capture completed!")
    print(f"📊 Captured {frame_count} frames in {elapsed_time:.1f} seconds")
    print(f"🎬 Actual FPS: {actual_fps:.1f}")
    
    return True

def test_camera_settings():
    """Test different camera settings"""
    
    print("\n🔧 Testing different camera settings...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera for settings test")
        return False
    
    # Test different resolutions
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080)
    ]
    
    for width, height in resolutions:
        print(f"📐 Testing resolution: {width}x{height}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read frame
        ret, frame = cap.read()
        if ret:
            actual_width = frame.shape[1]
            actual_height = frame.shape[0]
            print(f"  ✅ Frame captured: {actual_width}x{actual_height}")
        else:
            print(f"  ❌ Failed to capture frame")
    
    cap.release()
    return True

if __name__ == "__main__":
    print("🚀 Camera Test Script")
    print("=" * 40)
    
    # Test basic camera functionality
    if test_camera():
        print("\n✅ Camera test completed successfully!")
        
        # Test camera settings
        test_camera_settings()
        
        print("\n🎉 All camera tests passed!")
        print("Your camera is working correctly.")
        print("If you still don't see the camera window, check:")
        print("1. Window is not minimized or behind other windows")
        print("2. Display scaling settings")
        print("3. Graphics driver issues")
        
    else:
        print("\n❌ Camera test failed!")
        print("Check your camera permissions and drivers.")
