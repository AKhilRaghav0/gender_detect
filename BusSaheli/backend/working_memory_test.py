#!/usr/bin/env python3
"""
Working Memory Performance Test
Tests memory management with proper object handling
"""

import gc
import psutil
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryTracker:
    """Simple memory tracker for testing"""
    
    def __init__(self):
        self.resources: List[Any] = []
        self.frame_count = 0
        self.last_cleanup = time.time()
        
    def register(self, resource: Any):
        """Register a resource for tracking"""
        self.resources.append(resource)
        logger.debug(f"📝 Registered resource: {len(self.resources)} total")
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("🧹 Starting cleanup...")
        
        # Clear resources
        resources_cleared = len(self.resources)
        self.resources.clear()
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection freed {collected} objects")
        logger.info(f"🗑️ Cleared {resources_cleared} tracked resources")
        
        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"📊 Current memory usage: {memory_mb:.1f} MB")
        
        self.last_cleanup = time.time()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        self.frame_count += 1
        
        # Time-based cleanup (every 3 seconds for testing)
        if time.time() - self.last_cleanup > 3:
            return True
        
        # Frame-based cleanup (every 20 frames for testing)
        if self.frame_count % 20 == 0:
            return True
        
        # Memory-based cleanup
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 50:  # More than 50MB
            return True
        
        return False
    
    def auto_cleanup(self):
        """Perform automatic cleanup if needed"""
        if self.should_cleanup():
            self.cleanup()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = psutil.virtual_memory()
        
        return {
            "total_memory_gb": round(memory_info.total / (1024**3), 2),
            "used_memory_gb": round(memory_info.used / (1024**3), 2),
            "available_memory_gb": round(memory_info.available / (1024**3), 2),
            "memory_percent": round(memory_info.percent, 2),
            "python_memory_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "tracked_resources": len(self.resources),
            "frame_count": self.frame_count
        }

class ImageData:
    """Simple image data class for testing"""
    def __init__(self, width: int, height: int, channels: int = 3):
        self.width = width
        self.height = height
        self.channels = channels
        self.data = [0] * (width * height * channels)
        self.timestamp = time.time()
    
    def __del__(self):
        logger.debug("🗑️ ImageData destroyed")

class DetectionResult:
    """Simple detection result class for testing"""
    def __init__(self, bbox: List[int], gender: str, confidence: float, face_id: str):
        self.bbox = bbox
        self.gender = gender
        self.confidence = confidence
        self.face_id = face_id
        self.timestamp = time.time()
    
    def __del__(self):
        logger.debug("🗑️ DetectionResult destroyed")

def simulate_image_processing(memory_tracker: MemoryTracker, frame_count: int):
    """Simulate image processing with memory management"""
    
    # Simulate creating image data
    image_data = ImageData(640, 480, 3)
    memory_tracker.register(image_data)
    
    # Simulate processing
    results = []
    for i in range(5):  # Simulate 5 face detections
        face_data = DetectionResult(
            bbox=[i*100, i*50, (i+1)*100, (i+1)*50],
            gender='Male' if i % 2 == 0 else 'Female',
            confidence=0.8 + (i * 0.05),
            face_id=f"face_{frame_count}_{i}"
        )
        results.append(face_data)
        memory_tracker.register(face_data)
    
    return results

def test_memory_stability():
    """Test memory stability over time"""
    logger.info("🧠 Starting Memory Stability Test...")
    
    memory_tracker = MemoryTracker()
    
    # Get initial memory stats
    initial_stats = memory_tracker.get_memory_stats()
    logger.info(f"📊 Initial Memory: {initial_stats['python_memory_mb']:.1f} MB")
    logger.info(f"📊 Initial Resources: {initial_stats['tracked_resources']}")
    
    # Test for 50 iterations
    for i in range(50):
        # Simulate frame processing
        results = simulate_image_processing(memory_tracker, i)
        
        # Auto cleanup
        memory_tracker.auto_cleanup()
        
        # Log progress every 10 iterations
        if i % 10 == 0:
            stats = memory_tracker.get_memory_stats()
            memory_growth = stats['python_memory_mb'] - initial_stats['python_memory_mb']
            logger.info(f"📊 Iteration {i}: Memory: {stats['python_memory_mb']:.1f} MB (+{memory_growth:.1f} MB), Resources: {stats['tracked_resources']}")
            
            if memory_growth > 20:  # More than 20MB growth
                logger.warning(f"⚠️ High memory growth detected: +{memory_growth:.1f} MB")
    
    # Final cleanup
    memory_tracker.cleanup()
    
    # Final stats
    final_stats = memory_tracker.get_memory_stats()
    total_growth = final_stats['python_memory_mb'] - initial_stats['python_memory_mb']
    
    logger.info("=" * 50)
    logger.info("📊 FINAL MEMORY REPORT")
    logger.info("=" * 50)
    logger.info(f"Initial Memory: {initial_stats['python_memory_mb']:.1f} MB")
    logger.info(f"Final Memory: {final_stats['python_memory_mb']:.1f} MB")
    logger.info(f"Total Growth: +{total_growth:.1f} MB")
    logger.info(f"Tracked Resources: {final_stats['tracked_resources']}")
    logger.info(f"Total Iterations: {final_stats['frame_count']}")
    
    if total_growth < 10:
        logger.info("✅ MEMORY STABLE - No significant leaks detected!")
        return True
    else:
        logger.warning(f"⚠️ MEMORY LEAK DETECTED - Growth: +{total_growth:.1f} MB")
        return False

def test_cleanup_effectiveness():
    """Test cleanup effectiveness"""
    logger.info("🧹 Testing Cleanup Effectiveness...")
    
    memory_tracker = MemoryTracker()
    
    # Create many resources
    for i in range(30):
        resource = ImageData(100, 100, 3)
        memory_tracker.register(resource)
    
    initial_resources = len(memory_tracker.resources)
    initial_memory = memory_tracker.get_memory_stats()['python_memory_mb']
    logger.info(f"📊 Created {initial_resources} resources, Memory: {initial_memory:.1f} MB")
    
    # Force cleanup
    memory_tracker.cleanup()
    
    final_resources = len(memory_tracker.resources)
    final_memory = memory_tracker.get_memory_stats()['python_memory_mb']
    memory_freed = initial_memory - final_memory
    
    logger.info(f"📊 After cleanup: {final_resources} resources remaining")
    logger.info(f"📊 Memory freed: {memory_freed:.1f} MB")
    
    if final_resources == 0 and memory_freed > 0:
        logger.info("✅ Cleanup working effectively!")
        return True
    else:
        logger.warning("⚠️ Cleanup not working properly")
        return False

def test_performance_comparison():
    """Test performance with and without memory management"""
    logger.info("⚡ Testing Performance Comparison...")
    
    # Test WITHOUT memory management
    logger.info("📊 Testing WITHOUT memory management...")
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Simulate processing without cleanup
    for i in range(20):
        results = []
        for j in range(5):
            face_data = DetectionResult([j*10, j*5, (j+1)*10, (j+1)*5], 'Male', 0.8, f"face_{i}_{j}")
            results.append(face_data)
        # No cleanup - simulate memory leak
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    without_time = end_time - start_time
    without_memory_growth = end_memory - start_memory
    
    logger.info(f"📊 WITHOUT management: {without_time:.2f}s, Memory growth: +{without_memory_growth:.1f} MB")
    
    # Test WITH memory management
    logger.info("📊 Testing WITH memory management...")
    memory_tracker = MemoryTracker()
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    # Simulate processing with cleanup
    for i in range(20):
        results = simulate_image_processing(memory_tracker, i)
        memory_tracker.auto_cleanup()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    
    with_time = end_time - start_time
    with_memory_growth = end_memory - start_memory
    
    logger.info(f"📊 WITH management: {with_time:.2f}s, Memory growth: +{with_memory_growth:.1f} MB")
    
    # Calculate improvements
    time_improvement = ((without_time - with_time) / without_time) * 100
    memory_improvement = ((without_memory_growth - with_memory_growth) / without_memory_growth) * 100 if without_memory_growth > 0 else 100
    
    logger.info("=" * 50)
    logger.info("📊 PERFORMANCE COMPARISON")
    logger.info("=" * 50)
    logger.info(f"Time Improvement: {time_improvement:.1f}%")
    logger.info(f"Memory Improvement: {memory_improvement:.1f}%")
    
    return time_improvement > 0 and memory_improvement > 0

if __name__ == "__main__":
    print("🧠 Memory Management Performance Test")
    print("=" * 50)
    
    # Test 1: Memory Stability
    stability_test = test_memory_stability()
    
    print("\n" + "=" * 50)
    
    # Test 2: Cleanup Effectiveness
    cleanup_test = test_cleanup_effectiveness()
    
    print("\n" + "=" * 50)
    
    # Test 3: Performance Comparison
    performance_test = test_performance_comparison()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Memory Stability: {'✅ PASS' if stability_test else '❌ FAIL'}")
    print(f"Cleanup Effectiveness: {'✅ PASS' if cleanup_test else '❌ FAIL'}")
    print(f"Performance Improvement: {'✅ PASS' if performance_test else '❌ FAIL'}")
    
    if stability_test and cleanup_test and performance_test:
        print("🎉 ALL TESTS PASSED - Memory management working correctly!")
    else:
        print("⚠️ Some tests failed - Memory management needs improvement")
