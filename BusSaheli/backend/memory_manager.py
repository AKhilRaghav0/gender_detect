#!/usr/bin/env python3
"""
Memory Management System for Bus Saheli
Comprehensive memory leak prevention and resource management
"""

import gc
import psutil
import logging
import weakref
import threading
import time
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    python_memory: float
    timestamp: datetime

class ResourceTracker:
    """Tracks and manages resources to prevent memory leaks"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.weak_refs: Dict[str, weakref.ref] = {}
        self.cleanup_callbacks: Dict[str, callable] = {}
        self.lock = threading.Lock()
        
    def register(self, resource_id: str, resource: Any, cleanup_callback: Optional[callable] = None):
        """Register a resource for tracking"""
        with self.lock:
            self.resources[resource_id] = resource
            self.weak_refs[resource_id] = weakref.ref(resource)
            if cleanup_callback:
                self.cleanup_callbacks[resource_id] = cleanup_callback
            logger.debug(f"📝 Registered resource: {resource_id}")
    
    def unregister(self, resource_id: str):
        """Unregister and cleanup a resource"""
        with self.lock:
            if resource_id in self.resources:
                # Call cleanup callback if exists
                if resource_id in self.cleanup_callbacks:
                    try:
                        self.cleanup_callbacks[resource_id]()
                    except Exception as e:
                        logger.error(f"❌ Cleanup callback failed for {resource_id}: {e}")
                
                # Remove from tracking
                del self.resources[resource_id]
                if resource_id in self.weak_refs:
                    del self.weak_refs[resource_id]
                if resource_id in self.cleanup_callbacks:
                    del self.cleanup_callbacks[resource_id]
                
                logger.debug(f"🗑️ Unregistered resource: {resource_id}")
    
    def cleanup_dead_resources(self):
        """Clean up resources that are no longer referenced"""
        with self.lock:
            dead_resources = []
            for resource_id, weak_ref in self.weak_refs.items():
                if weak_ref() is None:  # Object was garbage collected
                    dead_resources.append(resource_id)
            
            for resource_id in dead_resources:
                self.unregister(resource_id)
                logger.debug(f"🧹 Cleaned up dead resource: {resource_id}")
    
    def get_resource_count(self) -> int:
        """Get number of tracked resources"""
        with self.lock:
            return len(self.resources)

class MemoryManager:
    """Comprehensive memory management system"""
    
    def __init__(self, max_memory_percent: float = 80.0, cleanup_interval: int = 100):
        self.max_memory_percent = max_memory_percent
        self.cleanup_interval = cleanup_interval
        self.resource_tracker = ResourceTracker()
        self.frame_count = 0
        self.last_cleanup = time.time()
        self.memory_history: List[MemoryStats] = []
        self.max_history = 1000
        
        # Memory thresholds
        self.warning_threshold = 70.0
        self.critical_threshold = 90.0
        
        logger.info("🧠 Memory Manager initialized")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = psutil.virtual_memory()
        
        return MemoryStats(
            total_memory=memory_info.total / (1024**3),  # GB
            available_memory=memory_info.available / (1024**3),  # GB
            used_memory=memory_info.used / (1024**3),  # GB
            memory_percent=memory_info.percent,
            python_memory=process.memory_info().rss / (1024**3),  # GB
            timestamp=datetime.now()
        )
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits"""
        stats = self.get_memory_stats()
        
        # Add to history
        self.memory_history.append(stats)
        if len(self.memory_history) > self.max_history:
            self.memory_history = self.memory_history[-self.max_history:]
        
        # Check thresholds
        if stats.memory_percent > self.critical_threshold:
            logger.error(f"🚨 CRITICAL: Memory usage at {stats.memory_percent:.1f}%")
            return False
        elif stats.memory_percent > self.warning_threshold:
            logger.warning(f"⚠️ WARNING: Memory usage at {stats.memory_percent:.1f}%")
        
        return True
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("🧹 Starting forced memory cleanup...")
        
        # Clean up dead resources
        self.resource_tracker.cleanup_dead_resources()
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection freed {collected} objects")
        
        # Clear memory history if too large
        if len(self.memory_history) > self.max_history * 0.8:
            self.memory_history = self.memory_history[-self.max_history//2:]
            logger.info("📊 Cleared old memory history")
        
        self.last_cleanup = time.time()
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        self.frame_count += 1
        
        # Time-based cleanup
        if time.time() - self.last_cleanup > 30:  # Every 30 seconds
            return True
        
        # Frame-based cleanup
        if self.frame_count % self.cleanup_interval == 0:
            return True
        
        # Memory-based cleanup
        if not self.check_memory_usage():
            return True
        
        return False
    
    def auto_cleanup(self):
        """Perform automatic cleanup if needed"""
        if self.should_cleanup():
            self.force_cleanup()
    
    @contextmanager
    def managed_resource(self, resource_id: str, resource: Any, cleanup_callback: Optional[callable] = None):
        """Context manager for automatic resource cleanup"""
        try:
            self.resource_tracker.register(resource_id, resource, cleanup_callback)
            yield resource
        finally:
            self.resource_tracker.unregister(resource_id)
    
    def cleanup_image_resources(self, *image_vars):
        """Clean up OpenCV image resources"""
        for var_name, image in image_vars:
            if image is not None:
                try:
                    # Clear image data
                    if hasattr(image, 'release'):
                        image.release()
                    del image
                    logger.debug(f"🗑️ Cleaned up image: {var_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to cleanup image {var_name}: {e}")
    
    def cleanup_detection_resources(self, results: List[Dict], faces: List[Any] = None):
        """Clean up detection-related resources"""
        try:
            # Clean up results
            if results:
                for result in results:
                    if 'face' in result and hasattr(result['face'], 'release'):
                        result['face'].release()
                results.clear()
            
            # Clean up faces
            if faces:
                for face in faces:
                    if hasattr(face, 'release'):
                        face.release()
                faces.clear()
            
            logger.debug("🗑️ Cleaned up detection resources")
        except Exception as e:
            logger.error(f"❌ Failed to cleanup detection resources: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        stats = self.get_memory_stats()
        
        return {
            "current_memory": {
                "total_gb": round(stats.total_memory, 2),
                "used_gb": round(stats.used_memory, 2),
                "available_gb": round(stats.available_memory, 2),
                "percent": round(stats.memory_percent, 2),
                "python_memory_gb": round(stats.python_memory, 2)
            },
            "tracked_resources": self.resource_tracker.get_resource_count(),
            "frame_count": self.frame_count,
            "last_cleanup": self.last_cleanup,
            "memory_trend": self._calculate_memory_trend(),
            "recommendations": self._get_memory_recommendations(stats)
        }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend"""
        if len(self.memory_history) < 10:
            return "insufficient_data"
        
        recent = self.memory_history[-10:]
        older = self.memory_history[-20:-10] if len(self.memory_history) >= 20 else self.memory_history[:-10]
        
        recent_avg = sum(s.memory_percent for s in recent) / len(recent)
        older_avg = sum(s.memory_percent for s in older) / len(older)
        
        if recent_avg > older_avg + 5:
            return "increasing"
        elif recent_avg < older_avg - 5:
            return "decreasing"
        else:
            return "stable"
    
    def _get_memory_recommendations(self, stats: MemoryStats) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        if stats.memory_percent > self.critical_threshold:
            recommendations.append("🚨 CRITICAL: Immediate cleanup required")
            recommendations.append("🔄 Consider reducing detection frequency")
            recommendations.append("🗑️ Clear cached data and restart if needed")
        elif stats.memory_percent > self.warning_threshold:
            recommendations.append("⚠️ High memory usage - consider cleanup")
            recommendations.append("📊 Monitor memory trend closely")
        
        if self.resource_tracker.get_resource_count() > 100:
            recommendations.append("🔧 Too many tracked resources - cleanup needed")
        
        if len(self.memory_history) > self.max_history * 0.9:
            recommendations.append("📈 Memory history full - clearing old data")
        
        return recommendations

# Global memory manager instance
memory_manager = MemoryManager()

# Convenience functions
def cleanup_image_resources(*image_vars):
    """Clean up image resources"""
    memory_manager.cleanup_image_resources(*image_vars)

def cleanup_detection_resources(results: List[Dict], faces: List[Any] = None):
    """Clean up detection resources"""
    memory_manager.cleanup_detection_resources(results, faces)

def auto_cleanup():
    """Perform automatic cleanup"""
    memory_manager.auto_cleanup()

def get_memory_report() -> Dict[str, Any]:
    """Get memory report"""
    return memory_manager.get_memory_report()

def managed_resource(resource_id: str, resource: Any, cleanup_callback: Optional[callable] = None):
    """Context manager for resource management"""
    return memory_manager.managed_resource(resource_id, resource, cleanup_callback)

# Example usage and testing
if __name__ == "__main__":
    import cv2
    import numpy as np
    
    # Test memory manager
    print("🧠 Testing Memory Manager...")
    
    # Get initial memory stats
    stats = memory_manager.get_memory_stats()
    print(f"Initial memory: {stats.memory_percent:.1f}%")
    
    # Simulate image processing
    for i in range(10):
        # Create test images
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with managed_resource(f"test_image_{i}", image):
            # Simulate processing
            processed = cv2.resize(image, (320, 240))
            result = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Auto cleanup
        auto_cleanup()
        
        if i % 5 == 0:
            report = get_memory_report()
            print(f"After {i+1} iterations: {report['current_memory']['percent']:.1f}%")
    
    # Final report
    final_report = get_memory_report()
    print("\n📊 Final Memory Report:")
    print(f"Memory usage: {final_report['current_memory']['percent']:.1f}%")
    print(f"Tracked resources: {final_report['tracked_resources']}")
    print(f"Memory trend: {final_report['memory_trend']}")
    print("Recommendations:")
    for rec in final_report['recommendations']:
        print(f"  - {rec}")
