#!/usr/bin/env python3
"""
Simple Memory Performance Test
Tests memory management without OpenCV dependencies
"""

import gc
import psutil
import time
import logging
import weakref
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMemoryManager:
    """Simplified memory manager for testing"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.weak_refs: Dict[str, weakref.ref] = {}
        self.cleanup_callbacks: Dict[str, callable] = {}
        self.frame_count = 0
        self.last_cleanup = time.time()
        
    def register(self, resource_id: str, resource: Any, cleanup_callback: Optional[callable] = None):
        """Register a resource for tracking"""
        self.resources[resource_id] = resource
        self.weak_refs[resource_id] = weakref.ref(resource)
        if cleanup_callback:
            self.cleanup_callbacks[resource_id] = cleanup_callback
        logger.debug(f"📝 Registered resource: {resource_id}")
    
    def unregister(self, resource_id: str):
        """Unregister and cleanup a resource"""
        if resource_id in self.resources:
            if resource_id in self.cleanup_callbacks:
                try:
                    self.cleanup_callbacks[resource_id]()
                except Exception as e:
                    logger.error(f"❌ Cleanup callback failed for {resource_id}: {e}")
            
            del self.resources[resource_id]
            if resource_id in self.weak_refs:
                del self.weak_refs[resource_id]
            if resource_id in self.cleanup_callbacks:
                del self.cleanup_callbacks[resource_id]
            
            logger.debug(f"🗑️ Unregistered resource: {resource_id}")
    
    def cleanup_dead_resources(self):
        """Clean up resources that are no longer referenced"""
        dead_resources = []
        for resource_id, weak_ref in self.weak_refs.items():
            if weak_ref() is None:
                dead_resources.append(resource_id)
        
        for resource_id in dead_resources:
            self.unregister(resource_id)
            logger.debug(f"🧹 Cleaned up dead resource: {resource_id}")
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("🧹 Starting forced memory cleanup...")
        
        # Clean up dead resources
        self.cleanup_dead_resources()
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection freed {collected} objects")
        
        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"📊 Current memory usage: {memory_mb:.1f} MB")
        
        self.last_cleanup = time.time()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        self.frame_count += 1
        
        # Time-based cleanup (every 5 seconds for testing)
        if time.time() - self.last_cleanup > 5:
            return True
        
        # Frame-based cleanup (every 50 frames for testing)
        if self.frame_count % 50 == 0:
            return True
        
        # Memory-based cleanup
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 100:  # More than 100MB
            return True
        
        return False
    
    def auto_cleanup(self):
        """Perform automatic cleanup if needed"""
        if self.should_cleanup():
            self.force_cleanup()
    
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

def simulate_image_processing(memory_manager: SimpleMemoryManager, frame_count: int):
    """Simulate image processing with memory management"""
    
    # Simulate creating image data
    image_data = {
        'width': 640,
        'height': 480,
        'channels': 3,
        'data': [0] * (640 * 480 * 3),  # Simulate image data
        'timestamp': time.time()
    }
    
    # Register for cleanup
    memory_manager.register(
        f"image_{frame_count}", 
        image_data,
        lambda: logger.debug(f"🗑️ Cleaned up image_{frame_count}")
    )
    
    # Simulate processing
    results = []
    for i in range(5):  # Simulate 5 face detections
        face_data = {
            'bbox': [i*100, i*50, (i+1)*100, (i+1)*50],
            'gender': 'Male' if i % 2 == 0 else 'Female',
            'confidence': 0.8 + (i * 0.05),
            'face_id': f"face_{frame_count}_{i}"
        }
        results.append(face_data)
    
    # Register results for cleanup
    memory_manager.register(
        f"results_{frame_count}",
        results,
        lambda: logger.debug(f"🗑️ Cleaned up results_{frame_count}")
    )
    
    return results

def test_memory_stability():
    """Test memory stability over time"""
    logger.info("🧠 Starting Memory Stability Test...")
    
    memory_manager = SimpleMemoryManager()
    
    # Get initial memory stats
    initial_stats = memory_manager.get_memory_stats()
    logger.info(f"📊 Initial Memory: {initial_stats['python_memory_mb']:.1f} MB")
    logger.info(f"📊 Initial Resources: {initial_stats['tracked_resources']}")
    
    # Test for 100 iterations
    for i in range(100):
        # Simulate frame processing
        results = simulate_image_processing(memory_manager, i)
        
        # Auto cleanup
        memory_manager.auto_cleanup()
        
        # Log progress every 20 iterations
        if i % 20 == 0:
            stats = memory_manager.get_memory_stats()
            memory_growth = stats['python_memory_mb'] - initial_stats['python_memory_mb']
            logger.info(f"📊 Iteration {i}: Memory: {stats['python_memory_mb']:.1f} MB (+{memory_growth:.1f} MB), Resources: {stats['tracked_resources']}")
            
            if memory_growth > 50:  # More than 50MB growth
                logger.warning(f"⚠️ High memory growth detected: +{memory_growth:.1f} MB")
    
    # Final stats
    final_stats = memory_manager.get_memory_stats()
    total_growth = final_stats['python_memory_mb'] - initial_stats['python_memory_mb']
    
    logger.info("=" * 50)
    logger.info("📊 FINAL MEMORY REPORT")
    logger.info("=" * 50)
    logger.info(f"Initial Memory: {initial_stats['python_memory_mb']:.1f} MB")
    logger.info(f"Final Memory: {final_stats['python_memory_mb']:.1f} MB")
    logger.info(f"Total Growth: +{total_growth:.1f} MB")
    logger.info(f"Tracked Resources: {final_stats['tracked_resources']}")
    logger.info(f"Total Iterations: {final_stats['frame_count']}")
    
    if total_growth < 20:
        logger.info("✅ MEMORY STABLE - No significant leaks detected!")
        return True
    else:
        logger.warning(f"⚠️ MEMORY LEAK DETECTED - Growth: +{total_growth:.1f} MB")
        return False

def test_cleanup_effectiveness():
    """Test cleanup effectiveness"""
    logger.info("🧹 Testing Cleanup Effectiveness...")
    
    memory_manager = SimpleMemoryManager()
    
    # Create many resources
    for i in range(50):
        resource = {'data': [0] * 1000, 'id': i}
        memory_manager.register(f"resource_{i}", resource)
    
    initial_resources = len(memory_manager.resources)
    logger.info(f"📊 Created {initial_resources} resources")
    
    # Force cleanup
    memory_manager.force_cleanup()
    
    final_resources = len(memory_manager.resources)
    cleaned_resources = initial_resources - final_resources
    
    logger.info(f"📊 After cleanup: {final_resources} resources remaining")
    logger.info(f"📊 Cleaned up: {cleaned_resources} resources")
    
    if cleaned_resources > 0:
        logger.info("✅ Cleanup working effectively!")
        return True
    else:
        logger.warning("⚠️ Cleanup not working properly")
        return False

if __name__ == "__main__":
    print("🧠 Memory Management Performance Test")
    print("=" * 50)
    
    # Test 1: Memory Stability
    stability_test = test_memory_stability()
    
    print("\n" + "=" * 50)
    
    # Test 2: Cleanup Effectiveness
    cleanup_test = test_cleanup_effectiveness()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Memory Stability: {'✅ PASS' if stability_test else '❌ FAIL'}")
    print(f"Cleanup Effectiveness: {'✅ PASS' if cleanup_test else '❌ FAIL'}")
    
    if stability_test and cleanup_test:
        print("🎉 ALL TESTS PASSED - Memory management working correctly!")
    else:
        print("⚠️ Some tests failed - Memory management needs improvement")
