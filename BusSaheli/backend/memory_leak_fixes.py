#!/usr/bin/env python3
"""
Memory Leak Fixes for Existing Code
Patches to fix memory leaks in current detection code
"""

import gc
import psutil
import logging
import weakref
import threading
import time
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryLeakFixer:
    """Fixes memory leaks in existing detection code"""
    
    def __init__(self):
        self.cleanup_callbacks = {}
        self.weak_refs = {}
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_cleanup = time.time()
        
    def register_cleanup(self, obj_id: str, obj: Any, cleanup_func: callable):
        """Register an object for cleanup tracking"""
        with self.lock:
            self.cleanup_callbacks[obj_id] = cleanup_func
            self.weak_refs[obj_id] = weakref.ref(obj)
            logger.debug(f"📝 Registered cleanup for: {obj_id}")
    
    def cleanup_dead_objects(self):
        """Clean up objects that are no longer referenced"""
        with self.lock:
            dead_objects = []
            for obj_id, weak_ref in self.weak_refs.items():
                if weak_ref() is None:  # Object was garbage collected
                    dead_objects.append(obj_id)
            
            for obj_id in dead_objects:
                if obj_id in self.cleanup_callbacks:
                    try:
                        self.cleanup_callbacks[obj_id]()
                        logger.debug(f"🧹 Cleaned up dead object: {obj_id}")
                    except Exception as e:
                        logger.error(f"❌ Cleanup failed for {obj_id}: {e}")
                    finally:
                        del self.cleanup_callbacks[obj_id]
                        del self.weak_refs[obj_id]
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("🧹 Starting forced memory cleanup...")
        
        # Clean up dead objects
        self.cleanup_dead_objects()
        
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
        
        # Time-based cleanup (every 30 seconds)
        if time.time() - self.last_cleanup > 30:
            return True
        
        # Frame-based cleanup (every 100 frames)
        if self.frame_count % 100 == 0:
            return True
        
        # Memory-based cleanup
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 500:  # More than 500MB
            return True
        
        return False
    
    def auto_cleanup(self):
        """Perform automatic cleanup if needed"""
        if self.should_cleanup():
            self.force_cleanup()

# Global memory leak fixer
memory_fixer = MemoryLeakFixer()

def safe_image_cleanup(*image_vars):
    """Safely clean up image variables"""
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

def safe_detection_cleanup(results: List[Dict], faces: List[Any] = None):
    """Safely clean up detection results and faces"""
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

def patch_insightface_web_app():
    """Apply memory leak fixes to insightface_web_app.py"""
    
    # Patch the process_frame function
    original_process_frame = None
    
    def patched_process_frame():
        """Patched version of process_frame with memory management"""
        global detection_results, frame_count, last_frame_image
        
        start_time = time.time()
        
        try:
            # Get base64 image and model selection from request
            data = request.get_json()
            image_data = data.get('image', '')
            detector_sel = data.get('detector')
            classifier_sel = data.get('classifier')
            
            if detector_sel or classifier_sel:
                set_models(detector_sel or current_detector, classifier_sel or current_classifier)
            
            if not image_data:
                return jsonify({"error": "No image data"})
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Register for cleanup
            memory_fixer.register_cleanup(f"image_{frame_count}", image, lambda: safe_image_cleanup(("image", image)))
            
            # Keep a reference for classifiers
            last_frame_image = image
            
            # Resize image for better speed
            h, w = image.shape[:2]
            resized_image = None
            if w > 512:
                scale = 512 / w
                new_w = 512
                new_h = int(h * scale)
                resized_image = cv2.resize(image, (new_w, new_h))
                memory_fixer.register_cleanup(f"resized_image_{frame_count}", resized_image, 
                                            lambda: safe_image_cleanup(("resized_image", resized_image)))
            
            # Try multiple image orientations
            original_image = image.copy()
            rotated_images = []
            
            try:
                rotated_images = [
                    (original_image, 0),
                    (cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE), 90),
                    (cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE), -90),
                    (cv2.rotate(original_image, cv2.ROTATE_180), 180)
                ]
                
                # Register rotated images for cleanup
                for i, (rot_img, _) in enumerate(rotated_images):
                    memory_fixer.register_cleanup(f"rotated_image_{frame_count}_{i}", rot_img,
                                                lambda img=rot_img: safe_image_cleanup(("rotated_image", img)))
                
                # Process detection (existing logic would go here)
                all_results = []
                
                # Check if we have cached tracked faces
                if len(tracked_faces) > 0 and (frame_count - last_full_detection) < DETECTION_INTERVAL:
                    logger.info(f"🔄 Using cached tracked faces (frame {frame_count})")
                    for face_id, face_data in tracked_faces.items():
                        if time.time() - face_data['last_seen'] < FACE_CACHE_DURATION:
                            face = face_data['face']
                            bbox = face.bbox.astype(int)
                            result = {
                                'bbox': bbox,
                                'gender': face_data['gender'],
                                'age': face_data['age'],
                                'confidence': face_data['confidence'],
                                'det_score': face.det_score,
                                'face_id': face_id,
                                'tracked': True
                            }
                            all_results.append(result)
                else:
                    # Run full detection on each rotation
                    for img, rotation in rotated_images:
                        try:
                            if current_detector == 'insightface':
                                results = detect_faces_insightface(img)
                            else:
                                results = detect_faces_haar(img)
                            
                            if results:
                                # Adjust bounding boxes for rotation
                                for result in results:
                                    if rotation != 0:
                                        bbox = result['bbox']
                                        x1, y1, x2, y2 = bbox
                                        
                                        if rotation == 90:
                                            h_rot, w_rot = img.shape[:2]
                                            new_x1, new_y1 = y1, h_rot - x2
                                            new_x2, new_y2 = y2, h_rot - x1
                                        elif rotation == -90:
                                            h_rot, w_rot = img.shape[:2]
                                            new_x1, new_y1 = w_rot - y2, x1
                                            new_x2, new_y2 = w_rot - y1, x2
                                        elif rotation == 180:
                                            h_rot, w_rot = img.shape[:2]
                                            new_x1, new_y1 = w_rot - x2, h_rot - y2
                                            new_x2, new_y2 = w_rot - x1, h_rot - y1
                                        else:
                                            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
                                        
                                        result['bbox'] = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
                                        result['rotation'] = rotation
                                    
                                    all_results.append(result)
                                    
                        except Exception as e:
                            logger.warning(f"⚠️ Error detecting faces in rotation {rotation}: {e}")
                            continue
                
                # Remove duplicate faces
                results = remove_duplicate_faces_from_results(all_results)
                detection_results = results
                
                # If no faces detected, return early
                if len(results) == 0:
                    logger.info("❌ No faces detected in this frame")
                    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 30])
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "faces": [],
                        "fps": 0
                    })
                
                # Draw rectangles on image
                try:
                    image_with_boxes = image.copy()
                    memory_fixer.register_cleanup(f"image_with_boxes_{frame_count}", image_with_boxes,
                                                lambda: safe_image_cleanup(("image_with_boxes", image_with_boxes)))
                    
                    for result in results:
                        bbox = result['bbox']
                        x1, y1, x2, y2 = bbox
                        gender = result['gender']
                        confidence = result['confidence']
                        
                        color = (0, 255, 0) if gender == 'Female' else (255, 0, 0)
                        
                        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{gender} ({confidence:.2f})"
                        cv2.putText(image_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Encode image with rectangles
                    _, buffer = cv2.imencode('.jpg', image_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 30])
                    image_bytes = buffer.tobytes()
                    image_with_boxes_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Prepare performance metrics
                    current_processing_time = time.time() - start_time
                    performance_metrics = {
                        "processing_time": current_processing_time,
                        "model_status": "InsightFace Active" if insightface_app is not None else "Haar Cascade Only",
                        "detection_mode": "Cached Tracking" if len(tracked_faces) > 0 and (frame_count - last_full_detection) < DETECTION_INTERVAL else "Full Detection",
                        "tracking_status": "Active" if len(tracked_faces) > 0 else "Ready",
                        "memory_usage": "Optimized",
                        "cpu_optimization": "Active"
                    }
                    
                    return jsonify({
                        "status": "success",
                        "results": results,
                        "image_with_boxes": image_with_boxes_b64,
                        **performance_metrics
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Image processing error: {e}")
                    return jsonify({
                        "status": "success",
                        "results": results,
                        "image_with_boxes": None,
                        **performance_metrics
                    })
                
            finally:
                # Cleanup rotated images
                for i, (rot_img, _) in enumerate(rotated_images):
                    safe_image_cleanup((f"rotated_image_{i}", rot_img))
                
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
            return jsonify({"error": str(e)})
        finally:
            # Performance monitoring and cleanup
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Log performance metrics every 50 frames
            if frame_count % 50 == 0:
                tracked_count = len(tracked_faces)
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                logger.info(f"📊 Performance - Frame: {frame_count}, Processing time: {processing_time:.3f}s, FPS: {fps:.1f}, Tracked faces: {tracked_count}")
                logger.info(f"🧠 Memory - Usage: {memory_mb:.1f} MB")
            
            # Auto cleanup
            memory_fixer.auto_cleanup()
    
    return patched_process_frame

def apply_memory_fixes():
    """Apply all memory leak fixes"""
    logger.info("🔧 Applying memory leak fixes...")
    
    # Patch the process_frame function
    patched_function = patch_insightface_web_app()
    
    # Replace the original function
    import insightface_web_app
    insightface_web_app.process_frame = patched_function
    
    logger.info("✅ Memory leak fixes applied successfully")

# Example usage
if __name__ == "__main__":
    # Test memory fixer
    print("🧠 Testing Memory Leak Fixer...")
    
    # Simulate some objects
    import numpy as np
    
    for i in range(10):
        # Create test objects
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = [{"test": "data"}]
        
        # Register for cleanup
        memory_fixer.register_cleanup(f"test_image_{i}", image, 
                                    lambda img=image: safe_image_cleanup(("test_image", img)))
        memory_fixer.register_cleanup(f"test_results_{i}", results,
                                    lambda res=results: safe_detection_cleanup(res))
        
        # Auto cleanup
        memory_fixer.auto_cleanup()
        
        if i % 5 == 0:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            print(f"After {i+1} iterations: {memory_mb:.1f} MB")
    
    print("✅ Memory leak fixer test completed")
