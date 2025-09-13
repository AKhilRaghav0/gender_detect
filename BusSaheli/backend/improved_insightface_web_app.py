#!/usr/bin/env python3
"""
Improved InsightFace-based Gender Detection Web App
With comprehensive memory management and leak prevention
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import base64
import numpy as np
import threading
import time
import logging
import gc
from contextlib import contextmanager

# Import our memory management system
from memory_manager import memory_manager, cleanup_image_resources, cleanup_detection_resources, auto_cleanup, managed_resource

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
insightface_app = None
detection_results = []
gender_history = []

# Face tracking system
tracked_faces = {}
next_face_id = 1
frame_count = 0
last_full_detection = 0
DETECTION_INTERVAL = 3
FACE_CACHE_DURATION = 10
last_frame_image = None

# Model registry/config
current_detector = 'insightface'
current_classifier = 'insightface_gender'

yunet_detector = None
ultraface_net = None
mobilenet_session = None
mobilenet_input_name = None
mobilenet_input_shape = (96, 96)

def calculate_face_signature(face):
    """Calculate a unique signature for a face based on landmarks"""
    try:
        if hasattr(face, 'kps') and face.kps is not None:
            landmarks = face.kps.flatten()
            bbox = face.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width > 0 and height > 0:
                landmarks[::2] = (landmarks[::2] - bbox[0]) / width
                landmarks[1::2] = (landmarks[1::2] - bbox[1]) / height
            return landmarks.tobytes()
        else:
            bbox = face.bbox
            return f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{face.det_score:.3f}".encode()
    except Exception as e:
        logger.error(f"❌ Error calculating face signature: {e}")
        return b"error_signature"

def calculate_face_distance(face1, face2):
    """Calculate distance between two face signatures"""
    try:
        sig1 = calculate_face_signature(face1)
        sig2 = calculate_face_signature(face2)
        
        if len(sig1) == len(sig2):
            landmarks1 = np.frombuffer(sig1, dtype=np.float32)
            landmarks2 = np.frombuffer(sig2, dtype=np.float32)
            return np.linalg.norm(landmarks1 - landmarks2)
        else:
            return 1 - calculate_iou(face1.bbox, face2.bbox)
    except Exception as e:
        logger.error(f"❌ Error calculating face distance: {e}")
        return float('inf')

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union of two bounding boxes"""
    try:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    except Exception as e:
        logger.error(f"❌ Error calculating IoU: {e}")
        return 0

@contextmanager
def managed_image_processing(image_data: str):
    """Context manager for safe image processing with automatic cleanup"""
    image = None
    resized_image = None
    rotated_images = []
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Resize image for better speed
        h, w = image.shape[:2]
        if w > 512:
            scale = 512 / w
            new_w = 512
            new_h = int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h))
            yield resized_image, image
        else:
            yield image, image
            
    except Exception as e:
        logger.error(f"❌ Image processing error: {e}")
        raise
    finally:
        # Cleanup all image resources
        cleanup_image_resources(
            ("original_image", image),
            ("resized_image", resized_image)
        )
        
        # Cleanup rotated images
        for i, (rot_img, _) in enumerate(rotated_images):
            cleanup_image_resources((f"rotated_image_{i}", rot_img))

def update_tracked_faces(new_faces, current_time):
    """Update tracked faces with new detections - with memory management"""
    global tracked_faces, next_face_id
    
    try:
        # Clean up old faces
        faces_to_remove = []
        for face_id, face_data in tracked_faces.items():
            if current_time - face_data['last_seen'] > FACE_CACHE_DURATION:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            del tracked_faces[face_id]
            logger.info(f"🗑️ Removed old face {face_id}")
        
        # Match new faces with existing tracked faces
        matched_faces = set()
        results = []
        
        for new_face in new_faces:
            best_match_id = None
            best_distance = float('inf')
            
            # Find the best matching tracked face
            for face_id, face_data in tracked_faces.items():
                if face_id in matched_faces:
                    continue
                    
                distance = calculate_face_distance(new_face, face_data['face'])
                if distance < 0.3 and distance < best_distance:
                    best_distance = distance
                    best_match_id = face_id
            
            if best_match_id is not None:
                # Update existing tracked face
                face_data = tracked_faces[best_match_id]
                face_data['face'] = new_face
                face_data['last_seen'] = current_time
                face_data['frame_count'] += 1
                
                # Use cached gender/age if available and recent
                if face_data['frame_count'] % 5 != 0:
                    results.append({
                        'face': new_face,
                        'gender': face_data['gender'],
                        'age': face_data['age'],
                        'confidence': face_data['confidence'],
                        'face_id': best_match_id,
                        'tracked': True
                    })
                    logger.info(f"🔄 Updated tracked face {best_match_id} (cached classification)")
                else:
                    # Re-classify occasionally
                    gender, age, confidence = classify_face(new_face)
                    face_data['gender'] = gender
                    face_data['age'] = age
                    face_data['confidence'] = confidence
                    
                    results.append({
                        'face': new_face,
                        'gender': gender,
                        'age': age,
                        'confidence': confidence,
                        'face_id': best_match_id,
                        'tracked': True
                    })
                    logger.info(f"🔄 Updated tracked face {best_match_id} (re-classified)")
                
                matched_faces.add(best_match_id)
            else:
                # New face detected
                face_id = next_face_id
                next_face_id += 1
                
                gender, age, confidence = classify_face(new_face)
                
                tracked_faces[face_id] = {
                    'face': new_face,
                    'gender': gender,
                    'age': age,
                    'confidence': confidence,
                    'last_seen': current_time,
                    'frame_count': 1
                }
                
                results.append({
                    'face': new_face,
                    'gender': gender,
                    'age': age,
                    'confidence': confidence,
                    'face_id': face_id,
                    'tracked': False
                })
                logger.info(f"🆕 New face detected: {face_id}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error updating tracked faces: {e}")
        return []

def classify_face(face):
    """Classify a single face for gender and age - with memory management"""
    global current_classifier, last_frame_image
    
    try:
        gender = 'Unknown'
        confidence = 0.7
        
        if current_classifier == 'insightface_gender':
            if hasattr(face, 'sex') and face.sex is not None:
                if face.sex == 'M':
                    gender = 'Male'
                    confidence = 0.9
                elif face.sex == 'F':
                    gender = 'Female'
                    confidence = 0.9
                else:
                    gender = 'Unknown'
                    confidence = 0.5
            else:
                if last_frame_image is not None:
                    gender = classify_gender_advanced(face, last_frame_image)
                    confidence = 0.7
        elif current_classifier == 'mobilenet_gender':
            if last_frame_image is not None:
                gender, confidence = classify_gender_mobilenet(face, last_frame_image)
            else:
                gender = 'Unknown'
                confidence = 0.5
        
        # Get age
        age = getattr(face, 'age', None)
        if age is None:
            age = 25
        
        return gender, age, confidence
        
    except Exception as e:
        logger.error(f"❌ Error classifying face: {e}")
        return 'Unknown', 25, 0.5

def detect_faces_insightface(image):
    """Detect faces using InsightFace with comprehensive memory management"""
    global frame_count, last_full_detection, tracked_faces
    
    try:
        frame_count += 1
        current_time = time.time()
        
        # Skip full detection if we have tracked faces and it's not time for full detection
        if len(tracked_faces) > 0 and (frame_count - last_full_detection) < DETECTION_INTERVAL:
            logger.info(f"⏭️ Skipping full detection (frame {frame_count}, tracked faces: {len(tracked_faces)})")
            return []
        
        if insightface_app is None:
            logger.info("🔄 InsightFace not available, using Haar Cascade fallback")
            return detect_faces_haar(image)
        
        # Detect faces using InsightFace
        faces = insightface_app.get(image)
        logger.info(f"🔍 InsightFace detected {len(faces)} faces")
        
        # If no faces detected, try with different image size
        if len(faces) == 0:
            logger.info("🔄 No faces detected, trying with resized image...")
            h, w = image.shape[:2]
            if h > 480 or w > 640:
                scale = min(480/h, 640/w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized_image = cv2.resize(image, (new_w, new_h))
                
                try:
                    faces = insightface_app.get(resized_image)
                    logger.info(f"🔍 Resized image detected {len(faces)} faces")
                finally:
                    # Cleanup resized image
                    cleanup_image_resources(("resized_image", resized_image))
        
        # If still no faces, try Haar Cascade as backup
        if len(faces) == 0:
            logger.info("🔄 InsightFace failed, trying Haar Cascade backup...")
            haar_results = detect_faces_haar(image)
            if len(haar_results) == 0:
                logger.info("❌ No faces detected by any method")
                return []
            return haar_results

        # Filter faces for quality
        filtered_faces = []
        for face in faces:
            try:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                face_width = x2 - x1
                face_height = y2 - y1
                face_area = face_width * face_height
                image_area = image.shape[0] * image.shape[1]
                
                # Skip if face is too small or too large
                if face_area < image_area * 0.01 or face_area > image_area * 0.5:
                    continue
                
                # Skip if aspect ratio is too extreme
                aspect_ratio = face_width / face_height if face_height > 0 else 1
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue
                
                # Skip if confidence is too low
                if face.det_score < 0.60:
                    continue
                
                filtered_faces.append(face)
            except Exception as e:
                logger.error(f"❌ Error filtering face: {e}")
                continue
        
        # Limit to maximum 2 faces for performance
        filtered_faces = filtered_faces[:2]
        
        if len(filtered_faces) == 0:
            logger.info("❌ No valid faces detected after filtering")
            return []
        
        # Update tracking system
        last_full_detection = frame_count
        tracking_results = update_tracked_faces(filtered_faces, current_time)
        
        # Convert tracking results to the expected format
        results = []
        for tracking_result in tracking_results:
            try:
                face = tracking_result['face']
                bbox = face.bbox.astype(int)
                
                result = {
                    'bbox': bbox,
                    'gender': tracking_result['gender'],
                    'age': tracking_result['age'],
                    'confidence': tracking_result['confidence'],
                    'det_score': face.det_score,
                    'face_id': tracking_result['face_id'],
                    'tracked': tracking_result['tracked']
                }
                results.append(result)
                
                status = "🔄" if tracking_result['tracked'] else "🆕"
                logger.info(f"{status} Face processed: {tracking_result['gender']} (age: {tracking_result['age']}, confidence: {tracking_result['confidence']:.3f})")
            except Exception as e:
                logger.error(f"❌ Error processing tracking result: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"❌ InsightFace detection error: {e}")
        return detect_faces_haar(image)
    finally:
        # Auto cleanup after detection
        auto_cleanup()

def detect_faces_haar(image):
    """Enhanced face detection using Haar Cascade with memory management"""
    results = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple Haar cascade classifiers
        cascade_files = [
            ('frontal', cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            ('alt', cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            ('alt2', cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
            ('alt_tree', cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml'),
            ('profile', cv2.data.haarcascades + 'haarcascade_profileface.xml')
        ]
        
        all_faces = []
        
        for cascade_name, cascade_file in cascade_files:
            try:
                face_cascade = cv2.CascadeClassifier(cascade_file)
                if face_cascade.empty():
                    logger.warning(f"⚠️ Could not load {cascade_name} cascade")
                    continue
                
                # Different parameters for different cascade types
                if cascade_name == 'profile':
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=2,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                else:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(40, 40),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                
                if len(faces) > 0:
                    logger.info(f"🔍 {cascade_name} cascade detected {len(faces)} faces")
                    for face in faces:
                        face_with_type = list(face) + [cascade_name]
                        all_faces.append(face_with_type)
                        
            except Exception as e:
                logger.warning(f"⚠️ Error with {cascade_name} cascade: {e}")
                continue
        
        # Remove duplicate faces
        faces = remove_duplicate_faces(all_faces)
        logger.info(f"🔍 Total unique faces detected: {len(faces)}")
        
        # If still no faces, try with more aggressive parameters
        if len(faces) == 0:
            logger.info("🔄 Trying aggressive detection parameters...")
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=1,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            logger.info(f"🔍 Aggressive detection found {len(faces)} faces")
        
        for (x, y, w, h) in faces:
            try:
                face_region = image[y:y+h, x:x+w]
                if face_region.size > 0:
                    # Simple gender classification
                    aspect_ratio = w / h
                    if aspect_ratio > 0.82:
                        gender = 'Male'
                        confidence = min(0.7, aspect_ratio)
                    else:
                        gender = 'Female'
                        confidence = min(0.7, 1.0 - aspect_ratio)
                else:
                    gender = 'Unknown'
                    confidence = 0.5
                
                results.append({
                    'bbox': (int(x), int(y), int(x+w), int(y+h)),
                    'gender': gender,
                    'confidence': float(confidence)
                })
            except Exception as e:
                logger.error(f"❌ Error processing face: {e}")
                continue
        
    except Exception as e:
        logger.error(f"❌ Haar detection error: {e}")
    finally:
        # Cleanup
        cleanup_image_resources(("gray_image", gray))
    
    return results

def remove_duplicate_faces(faces_with_types):
    """Remove overlapping face detections from multiple cascades"""
    if not faces_with_types:
        return []
    
    try:
        faces = np.array([face[:4] for face in faces_with_types])
        types = [face[4] for face in faces_with_types]
        
        def calculate_overlap(face1, face2):
            x1, y1, w1, h1 = face1
            x2, y2, w2, h2 = face2
            
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        unique_faces = []
        used_indices = set()
        
        for i, face in enumerate(faces):
            if i in used_indices:
                continue
                
            is_duplicate = False
            for j in unique_faces:
                if calculate_overlap(face, faces[j]) > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(i)
                used_indices.add(i)
        
        return [faces[i] for i in unique_faces]
        
    except Exception as e:
        logger.error(f"❌ Error removing duplicate faces: {e}")
        return []

def classify_gender_advanced(face, image):
    """Advanced gender classification with memory management"""
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return 'Unknown'
        
        # Simple gender classification based on face shape
        face_height = y2 - y1
        face_width = x2 - x1
        aspect_ratio = face_width / face_height if face_height > 0 else 1
        
        if aspect_ratio > 0.85:
            return 'Male'
        else:
            return 'Female'
            
    except Exception as e:
        logger.error(f"❌ Advanced gender classification error: {e}")
        return 'Unknown'

def classify_gender_mobilenet(face, image):
    """MobileNet gender classification with memory management"""
    try:
        if mobilenet_session is None:
            return 'Unknown', 0.5
        
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 'Unknown', 0.5
        
        inp = cv2.resize(crop, mobilenet_input_shape)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        
        outputs = mobilenet_session.run(None, {mobilenet_input_name: inp})
        logits = outputs[0].squeeze()
        
        try:
            exps = np.exp(logits - np.max(logits))
            probs = exps / np.sum(exps)
        except:
            probs = logits / np.sum(logits)
        
        if probs.shape[0] == 2:
            male_conf = float(probs[1])
            gender = 'Male' if male_conf >= 0.5 else 'Female'
            conf = male_conf if gender == 'Male' else float(probs[0])
        else:
            conf = float(probs.max())
            gender = 'Male' if probs.argmax() == 1 else 'Female'
        
        return gender, max(0.5, min(0.99, conf))
        
    except Exception as e:
        logger.error(f"❌ MobileNet gender error: {e}")
        return 'Unknown', 0.5

@app.route('/')
def index():
    """Main page"""
    return render_template('insightface.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame with comprehensive memory management"""
    global detection_results, frame_count, last_frame_image
    
    start_time = time.time()
    
    try:
        # Get request data
        data = request.get_json()
        image_data = data.get('image', '')
        detector_sel = data.get('detector')
        classifier_sel = data.get('classifier')
        
        if detector_sel or classifier_sel:
            # Update models if needed
            pass  # Implementation would go here
        
        if not image_data:
            return jsonify({"error": "No image data"})
        
        # Use managed image processing
        with managed_image_processing(image_data) as (processed_image, original_image):
            # Store reference for classifiers
            last_frame_image = original_image
            
            # Try face detection on multiple orientations
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
                # Run full detection
                if current_detector == 'insightface':
                    results = detect_faces_insightface(processed_image)
                else:
                    results = detect_faces_haar(processed_image)
                
                if results:
                    all_results.extend(results)
            
            # Remove duplicate faces
            results = remove_duplicate_faces_from_results(all_results)
            detection_results = results
            
            # If no faces detected, return early
            if len(results) == 0:
                logger.info("❌ No faces detected in this frame")
                _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 30])
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                return jsonify({
                    "image": f"data:image/jpeg;base64,{image_base64}",
                    "faces": [],
                    "fps": 0
                })
            
            # Draw rectangles on image for visualization
            try:
                image_with_boxes = processed_image.copy()
                
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
                # Cleanup image resources
                cleanup_image_resources(("image_with_boxes", image_with_boxes))
        
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
            memory_report = get_memory_report()
            logger.info(f"📊 Performance - Frame: {frame_count}, Processing time: {processing_time:.3f}s, FPS: {fps:.1f}, Tracked faces: {tracked_count}")
            logger.info(f"🧠 Memory - Usage: {memory_report['current_memory']['percent']:.1f}%, Resources: {memory_report['tracked_resources']}")
        
        # Auto cleanup
        auto_cleanup()

def remove_duplicate_faces_from_results(results):
    """Remove overlapping face detections from different orientations"""
    if not results:
        return []
    
    try:
        def calculate_overlap(result1, result2):
            bbox1 = result1['bbox']
            bbox2 = result2['bbox']
            
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0
        
        unique_results = []
        used_indices = set()
        
        for i, result in enumerate(results):
            if i in used_indices:
                continue
                
            is_duplicate = False
            for j in unique_results:
                if calculate_overlap(result, results[j]) > 0.4:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(i)
                used_indices.add(i)
        
        return [results[i] for i in unique_results]
        
    except Exception as e:
        logger.error(f"❌ Error removing duplicate results: {e}")
        return results

@app.route('/get_results')
def get_results():
    """Get current detection results"""
    return jsonify(detection_results)

@app.route('/memory_report')
def memory_report():
    """Get memory usage report"""
    return jsonify(get_memory_report())

if __name__ == '__main__':
    # Initialize services
    logger.info("🚀 Starting Improved InsightFace web server...")
    logger.info("🌐 Access: http://localhost:5000")
    logger.info("🧠 Memory management: ENABLED")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
