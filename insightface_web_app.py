#!/usr/bin/env python3
"""
InsightFace-based Gender Detection Web App
Browser camera + InsightFace for accurate gender detection
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import base64
import numpy as np
import threading
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
insightface_app = None
detection_results = []
gender_history = []  # Store recent gender predictions for validation

# Face tracking system
tracked_faces = {}  # Dictionary to store tracked faces: {face_id: face_data}
next_face_id = 1
frame_count = 0
last_full_detection = 0
DETECTION_INTERVAL = 3  # Process full detection every 3 frames
FACE_CACHE_DURATION = 10  # Cache faces for 10 seconds
last_frame_image = None  # For classifiers needing the source frame

# Model registry/config
current_detector = 'insightface'  # 'insightface' | 'yunet' | 'ultraface'
current_classifier = 'insightface_gender'  # 'insightface_gender' | 'mobilenet_gender'

yunet_detector = None
ultraface_net = None
mobilenet_session = None
mobilenet_input_name = None
mobilenet_input_shape = (96, 96)

def calculate_face_signature(face):
    """Calculate a unique signature for a face based on landmarks"""
    if hasattr(face, 'kps') and face.kps is not None:
        # Use facial landmarks to create a signature
        landmarks = face.kps.flatten()
        # Normalize landmarks to make them scale-invariant
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width > 0 and height > 0:
            landmarks[::2] = (landmarks[::2] - bbox[0]) / width  # Normalize x coordinates
            landmarks[1::2] = (landmarks[1::2] - bbox[1]) / height  # Normalize y coordinates
        return landmarks.tobytes()
    else:
        # Fallback: use bounding box and detection score
        bbox = face.bbox
        return f"{bbox[0]:.2f}_{bbox[1]:.2f}_{bbox[2]:.2f}_{bbox[3]:.2f}_{face.det_score:.3f}".encode()

def calculate_face_distance(face1, face2):
    """Calculate distance between two face signatures"""
    try:
        sig1 = calculate_face_signature(face1)
        sig2 = calculate_face_signature(face2)
        
        if len(sig1) == len(sig2):
            # For landmark-based signatures
            landmarks1 = np.frombuffer(sig1, dtype=np.float32)
            landmarks2 = np.frombuffer(sig2, dtype=np.float32)
            return np.linalg.norm(landmarks1 - landmarks2)
        else:
            # For bbox-based signatures, use IoU
            bbox1 = face1.bbox
            bbox2 = face2.bbox
            return 1 - calculate_iou(bbox1, bbox2)
    except:
        # Fallback: use bounding box IoU
        return 1 - calculate_iou(face1.bbox, face2.bbox)

def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union of two bounding boxes"""
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

def update_tracked_faces(new_faces, current_time):
    """Update tracked faces with new detections"""
    global tracked_faces, next_face_id
    
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
            if distance < 0.3 and distance < best_distance:  # Threshold for matching
                best_distance = distance
                best_match_id = face_id
        
        if best_match_id is not None:
            # Update existing tracked face
            face_data = tracked_faces[best_match_id]
            face_data['face'] = new_face
            face_data['last_seen'] = current_time
            face_data['frame_count'] += 1
            
            # Use cached gender/age if available and recent
            if face_data['frame_count'] % 5 != 0:  # Skip classification every 5th frame
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

def classify_face(face):
    """Classify a single face for gender and age"""
    global current_classifier, last_frame_image
    gender = 'Unknown'
    confidence = 0.7
    # Route by classifier
    if current_classifier == 'insightface_gender':
        if hasattr(face, 'sex') and face.sex is not None:
            if face.sex == 'M':
                gender = 'Male'; confidence = 0.9
            elif face.sex == 'F':
                gender = 'Female'; confidence = 0.9
            else:
                gender = 'Unknown'; confidence = 0.5
        else:
            # Heuristic fallback requires image
            if last_frame_image is not None:
                gender = classify_gender_advanced(face, last_frame_image)
                confidence = 0.7
    elif current_classifier == 'mobilenet_gender':
        if last_frame_image is not None:
            gender, confidence = classify_gender_mobilenet(face, last_frame_image)
        else:
            gender = 'Unknown'; confidence = 0.5
    
    # Get age
    age = getattr(face, 'age', None)
    if age is None:
        age = 25  # Default age if not available
    
    return gender, age, confidence

def initialize_insightface():
    """Initialize InsightFace with better error handling"""
    global insightface_app
    try:
        import insightface
        logger.info("🔧 Initializing InsightFace...")
        
        # Try different model configurations
        try:
            # First try with buffalo_s (smallest)
            providers = ['CPUExecutionProvider']
            insightface_app = insightface.app.FaceAnalysis(
                name='buffalo_s',
                providers=providers
            )
            insightface_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
            logger.info("✅ InsightFace initialized with buffalo_s model")
        except Exception as e1:
            logger.warning(f"⚠️ buffalo_s failed: {e1}")
            try:
                # Try without specifying model name
                insightface_app = insightface.app.FaceAnalysis(providers=providers)
                insightface_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
                logger.info("✅ InsightFace initialized with default model")
            except Exception as e2:
                logger.warning(f"⚠️ Default model failed: {e2}")
                # Try with smaller detection size
                insightface_app = insightface.app.FaceAnalysis(providers=providers)
                insightface_app.prepare(ctx_id=0, det_size=(320, 320))
                logger.info("✅ InsightFace initialized with 320x320 detection")
                
    except Exception as e:
        logger.error(f"❌ Failed to initialize InsightFace: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        logger.error(f"❌ Error details: {str(e)}")
        logger.info("🔄 Falling back to Haar Cascade only...")
        insightface_app = None

def ensure_models_dir():
    import os
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    return models_dir

def download_file(url, dst_path):
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dst_path)
        logger.info(f"⬇️ Downloaded model: {dst_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not download {url}: {e}")
        return False

def initialize_yunet():
    """Initialize YuNet face detector (OpenCV Zoo)"""
    global yunet_detector
    try:
        models_dir = ensure_models_dir()
        yunet_path = os.path.join(models_dir, 'face_detection_yunet_2023mar.onnx')
        if not os.path.isfile(yunet_path):
            # OpenCV Zoo model URL
            download_file('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx', yunet_path)
        input_size = (320, 320)
        # API signatures vary across OpenCV versions
        if hasattr(cv2, 'FaceDetectorYN_create'):
            yunet_detector = cv2.FaceDetectorYN_create(yunet_path, '', input_size, 0.6, 0.3, 5000)
        elif hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2.FaceDetectorYN, 'create'):
            yunet_detector = cv2.FaceDetectorYN.create(yunet_path, '', input_size, 0.6, 0.3, 5000)
        else:
            yunet_detector = None
            logger.warning('⚠️ YuNet API not available in current OpenCV build')
        if yunet_detector is not None:
            logger.info('✅ YuNet initialized')
    except Exception as e:
        logger.error(f"❌ Failed to initialize YuNet: {e}")
        yunet_detector = None

def initialize_mobilenet_gender():
    """Initialize MobileNet ONNX gender classifier"""
    global mobilenet_session, mobilenet_input_name
    try:
        import onnxruntime as ort
        models_dir = ensure_models_dir()
        model_path = os.path.join(models_dir, 'gender_mobilenet_v2_0.5.onnx')
        if not os.path.isfile(model_path):
            # Public lightweight gender classifier (example URL placeholder)
            download_file('https://github.com/onnx/models/raw/main/vision/body_analysis/age_gender/models/gender_googlenet.onnx', model_path)
        providers = ['CPUExecutionProvider']
        mobilenet_session = ort.InferenceSession(model_path, providers=providers)
        mobilenet_input_name = mobilenet_session.get_inputs()[0].name
        logger.info('✅ MobileNet gender classifier initialized')
    except Exception as e:
        logger.error(f"❌ Failed to initialize MobileNet gender: {e}")
        mobilenet_session = None

def set_models(detector: str, classifier: str):
    """Set current models and ensure they are initialized"""
    global current_detector, current_classifier
    current_detector = detector
    current_classifier = classifier
    if detector == 'insightface' and insightface_app is None:
        initialize_insightface()
    if detector == 'yunet' and yunet_detector is None:
        initialize_yunet()
    if classifier == 'mobilenet_gender' and mobilenet_session is None:
        initialize_mobilenet_gender()

def detect_faces_insightface(image):
    """Detect faces and classify gender using InsightFace with tracking optimization"""
    global frame_count, last_full_detection, tracked_faces
    
    frame_count += 1
    current_time = time.time()
    
    # Skip full detection if we have tracked faces and it's not time for full detection
    if len(tracked_faces) > 0 and (frame_count - last_full_detection) < DETECTION_INTERVAL:
        logger.info(f"⏭️ Skipping full detection (frame {frame_count}, tracked faces: {len(tracked_faces)})")
        # Return empty results to use cached tracked faces
        return []
    
    if insightface_app is None:
        logger.info("🔄 InsightFace not available, using Haar Cascade fallback")
        return detect_faces_haar(image)
    
    try:
        # Detect faces using InsightFace
        faces = insightface_app.get(image)
        logger.info(f"🔍 InsightFace detected {len(faces)} faces")
        
        # If no faces detected, try with different image size
        if len(faces) == 0:
            logger.info("🔄 No faces detected, trying with resized image...")
            # Resize image for better detection
            h, w = image.shape[:2]
            if h > 480 or w > 640:
                scale = min(480/h, 640/w)
                new_h, new_w = int(h * scale), int(w * scale)
                resized_image = cv2.resize(image, (new_w, new_h))
                faces = insightface_app.get(resized_image)
                logger.info(f"🔍 Resized image detected {len(faces)} faces")
        
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
            # Extract face bounding box
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Filter out very small or very large detections (likely false positives)
            face_width = x2 - x1
            face_height = y2 - y1
            face_area = face_width * face_height
            image_area = image.shape[0] * image.shape[1]
            
            # Skip if face is too small (< 1% of image) or too large (> 50% of image)
            if face_area < image_area * 0.01 or face_area > image_area * 0.5:
                continue
            
            # Skip if aspect ratio is too extreme (likely not a face)
            aspect_ratio = face_width / face_height if face_height > 0 else 1
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Skip if confidence is too low (slightly relaxed to avoid misses)
            if face.det_score < 0.60:
                continue
            
            filtered_faces.append(face)
        
        # Limit to maximum 2 faces for Pi 5 performance
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
        logger.error(f"❌ InsightFace detection error: {e}")
        # Fallback to Haar Cascade
        return detect_faces_haar(image)
    
    return results

def detect_faces_yunet(image):
    """Detect faces using YuNet and return result dicts (without tracking)"""
    global yunet_detector
    if yunet_detector is None:
        initialize_yunet()
    if yunet_detector is None:
        return []
    h, w = image.shape[:2]
    try:
        # Update input size to current frame
        if hasattr(yunet_detector, 'setInputSize'):
            yunet_detector.setInputSize((w, h))
        faces = None
        if hasattr(yunet_detector, 'detect'):
            _, faces = yunet_detector.detect(image)
        if faces is None or len(faces) == 0:
            return []
        results = []
        for f in faces:
            # f: [x, y, w, h, score, l0x, l0y, l1x, l1y, ... l4x, l4y]
            x, y, ww, hh = f[:4]
            score = float(f[4])
            landmarks = np.array([[f[5], f[6]], [f[7], f[8]], [f[9], f[10]], [f[11], f[12]], [f[13], f[14]]], dtype=np.float32)
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(w, x + ww))
            y2 = int(min(h, y + hh))
            # Build MockFace with landmarks to leverage tracking and classifiers
            class MockFace:
                def __init__(self, bbox, kps, det_score):
                    self.bbox = np.array(bbox)
                    self.kps = kps
                    self.sex = None
                    self.age = None
                    self.det_score = det_score
            mock = MockFace([x1, y1, x2, y2], landmarks, score)
            # Classification will be handled later via tracker update
            results.append({'mock_face': mock})
        return results
    except Exception as e:
        logger.error(f"❌ YuNet detection error: {e}")
        return []

def classify_gender_mobilenet(face, image):
    """MobileNet ONNX gender classifier on face crop; returns (gender, confidence)"""
    global mobilenet_session
    try:
        if mobilenet_session is None:
            initialize_mobilenet_gender()
        if mobilenet_session is None:
            return 'Unknown', 0.5
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 'Unknown', 0.5
        inp = cv2.resize(crop, mobilenet_input_shape)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW
        outputs = mobilenet_session.run(None, {mobilenet_input_name: inp})
        logits = outputs[0].squeeze()
        # Assume 2-class [female, male] or [male, female]; robust mapping
        probs = None
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
            # Fallback binary
            conf = float(probs.max())
            gender = 'Male' if probs.argmax() == 1 else 'Female'
        return gender, max(0.5, min(0.99, conf))
    except Exception as e:
        logger.error(f"❌ MobileNet gender error: {e}")
        return 'Unknown', 0.5

def remove_duplicate_faces(faces_with_types):
    """Remove overlapping face detections from multiple cascades"""
    if not faces_with_types:
        return []
    
    # Convert to numpy array for easier processing
    faces = np.array([face[:4] for face in faces_with_types])  # x, y, w, h
    types = [face[4] for face in faces_with_types]  # cascade type
    
    # Calculate overlap between faces
    def calculate_overlap(face1, face2):
        x1, y1, w1, h1 = face1
        x2, y2, w2, h2 = face2
        
        # Calculate intersection
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
    
    # Keep faces with minimal overlap
    unique_faces = []
    used_indices = set()
    
    for i, face in enumerate(faces):
        if i in used_indices:
            continue
            
        # Check overlap with already selected faces
        is_duplicate = False
        for j in unique_faces:
            if calculate_overlap(face, faces[j]) > 0.3:  # 30% overlap threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_faces.append(i)
            used_indices.add(i)
    
    # Return unique faces without cascade type
    return [faces[i] for i in unique_faces]

def remove_duplicate_faces_from_results(results):
    """Remove overlapping face detections from multiple orientations"""
    if not results:
        return []
    
    # Calculate overlap between face results
    def calculate_overlap(result1, result2):
        bbox1 = result1['bbox']
        bbox2 = result2['bbox']
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
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
    
    # Keep faces with minimal overlap
    unique_results = []
    used_indices = set()
    
    for i, result in enumerate(results):
        if i in used_indices:
            continue
            
        # Check overlap with already selected faces
        is_duplicate = False
        for j in unique_results:
            if calculate_overlap(result, results[j]) > 0.4:  # 40% overlap threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_results.append(i)
            used_indices.add(i)
    
    # Return unique results
    return [results[i] for i in unique_results]

def detect_faces_haar(image):
    """Enhanced face detection using multiple Haar Cascade classifiers for different angles"""
    results = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple Haar cascade classifiers for different face angles
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
                    # Profile faces need different parameters
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=2,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                else:
                    # Frontal faces
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(40, 40),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                
                if len(faces) > 0:
                    logger.info(f"🔍 {cascade_name} cascade detected {len(faces)} faces")
                    # Add cascade type to faces for debugging
                    for face in faces:
                        face_with_type = list(face) + [cascade_name]
                        all_faces.append(face_with_type)
                        
            except Exception as e:
                logger.warning(f"⚠️ Error with {cascade_name} cascade: {e}")
                continue
        
        # Remove duplicate faces (overlapping detections)
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
            # Use advanced gender classification
            face_region = image[y:y+h, x:x+w]
            if face_region.size > 0:
                # Create a mock face object for advanced classification
                class MockFace:
                    def __init__(self, bbox):
                        self.bbox = np.array(bbox)
                        self.kps = None
                        self.sex = None
                        self.age = None
                        self.det_score = 0.8
                
                mock_face = MockFace([x, y, x+w, y+h])
                gender = classify_gender_advanced(mock_face, image)
                confidence = 0.7
            else:
                # Fallback to simple heuristic
                aspect_ratio = w / h
                if aspect_ratio > 0.85:
                    gender = 'Male'
                    confidence = min(0.7, aspect_ratio)
                else:
                    gender = 'Female'
                    confidence = min(0.7, 1.0 - aspect_ratio)
            
            results.append({
                'bbox': (int(x), int(y), int(x+w), int(y+h)),
                'gender': gender,
                'confidence': float(confidence)
            })
    
    except Exception as e:
        logger.error(f"❌ Haar detection error: {e}")
    
    return results

def classify_gender_advanced(face, image):
    """Simple but effective gender classification using key facial features"""
    logger.info("🔍 Starting advanced gender classification...")
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        face_region = image[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return 'Unknown'
        
        # Analyze face shape
        face_height = y2 - y1
        face_width = x2 - x1
        aspect_ratio = face_width / face_height if face_height > 0 else 1
        
        # Initialize gender scores
        male_score = 0
        female_score = 0
        
        # 1. Face Shape Analysis - More balanced scoring
        if aspect_ratio > 0.90:  # Very wide/square face (masculine)
            male_score += 2
        elif aspect_ratio < 0.70:  # Very narrow/oval face (feminine)
            female_score += 2
        elif aspect_ratio > 0.85:  # Moderately wide
            male_score += 1
        elif aspect_ratio < 0.80:  # Moderately narrow
            female_score += 1
        
        # 2. Simple Landmark Analysis
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            landmarks = face.kps
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Eye spacing analysis
            eye_distance = np.linalg.norm(right_eye - left_eye)
            if eye_distance > 0:
                eye_face_ratio = eye_distance / face_width
                
                # Males typically have wider eye spacing
                if eye_face_ratio > 0.35:
                    male_score += 1
                elif eye_face_ratio < 0.25:
                    female_score += 1
            
            # Jaw length analysis
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            jaw_length = np.linalg.norm(mouth_center - eye_center)
            
            if eye_distance > 0:
                jaw_ratio = jaw_length / eye_distance
                # Males typically have longer jaws
                if jaw_ratio > 0.75:
                    male_score += 1
                elif jaw_ratio < 0.55:
                    female_score += 1
        
        # 3. Skin Texture and Facial Hair Analysis
        try:
            # Convert to grayscale for texture analysis
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture smoothness (variance of Laplacian)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            # Females typically have smoother skin texture
            if laplacian_var < 100:  # Smoother texture
                female_score += 1
            elif laplacian_var > 200:  # Rougher texture (potential facial hair)
                male_score += 1
            
            # Enhanced facial hair detection
            # Look for mustache and beard areas
            if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
                nose = face.kps[2]
                left_mouth = face.kps[3]
                right_mouth = face.kps[4]
                
                # Mustache area (between nose and mouth)
                mustache_y1 = int(nose[1] + (left_mouth[1] - nose[1]) * 0.3)
                mustache_y2 = int(nose[1] + (left_mouth[1] - nose[1]) * 0.7)
                mustache_x1 = int(left_mouth[0])
                mustache_x2 = int(right_mouth[0])
                
                if (mustache_y1 < mustache_y2 and mustache_x1 < mustache_x2 and 
                    mustache_y1 >= 0 and mustache_y2 < face_region.shape[0] and
                    mustache_x1 >= 0 and mustache_x2 < face_region.shape[1]):
                    
                    mustache_region = face_region[mustache_y1:mustache_y2, mustache_x1:mustache_x2]
                    if mustache_region.size > 0:
                        gray_mustache = cv2.cvtColor(mustache_region, cv2.COLOR_BGR2GRAY)
                        
                        # Look for dark, coarse texture (facial hair)
                        dark_pixels = np.sum(gray_mustache < 80)
                        total_pixels = mustache_region.shape[0] * mustache_region.shape[1]
                        
                        if total_pixels > 0:
                            dark_ratio = dark_pixels / total_pixels
                            if dark_ratio > 0.3:  # Significant dark areas (facial hair)
                                male_score += 1
                            elif dark_ratio < 0.1:  # Very smooth area (no facial hair)
                                female_score += 1
        except:
            pass
        
        # 4. Enhanced Hair Analysis
        try:
            # Look at upper portion of face for hair analysis
            hair_region_y1 = max(0, y1 - int(face_height * 0.4))
            hair_region = image[hair_region_y1:y1, x1:x2]
            
            if hair_region.size > 0:
                # Convert to HSV for better hair detection
                hsv_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
                
                # Look for dark hair (typical for both genders but different patterns)
                dark_pixels = np.sum((hsv_hair[:,:,2] < 100) & (hsv_hair[:,:,1] > 30))
                total_pixels = hair_region.shape[0] * hair_region.shape[1]
                
                if total_pixels > 0:
                    dark_ratio = dark_pixels / total_pixels
                    
                    # Analyze hair distribution patterns
                    # Females often have more visible hair in the forehead area
                    if dark_ratio > 0.4:  # High hair visibility
                        female_score += 1
                    elif dark_ratio < 0.1:  # Very little hair visible (short hair)
                        male_score += 1
                
                # Analyze hair texture (longer hair tends to have different patterns)
                gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
                
                # Look for hair texture patterns
                # Longer hair (female) tends to have more flowing patterns
                # Shorter hair (male) tends to have more uniform patterns
                hair_edges = cv2.Canny(gray_hair, 50, 150)
                edge_density = np.sum(hair_edges > 0) / total_pixels
                
                if edge_density > 0.15:  # High edge density (longer, flowing hair)
                    female_score += 1
                elif edge_density < 0.05:  # Low edge density (short, uniform hair)
                    male_score += 1
        except:
            pass
        
        # 5. Enhanced Eyebrow and Eye Analysis
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            try:
                left_eye = face.kps[0]
                right_eye = face.kps[1]
                
                # Look at eyebrow region
                eyebrow_y = max(0, int(left_eye[1] - face_height * 0.15))
                eyebrow_region = image[eyebrow_y:int(left_eye[1]), x1:x2]
                
                if eyebrow_region.size > 0:
                    gray_eyebrow = cv2.cvtColor(eyebrow_region, cv2.COLOR_BGR2GRAY)
                    
                    # Thicker eyebrows are more masculine
                    eyebrow_thickness = np.mean(gray_eyebrow < 100)  # Dark pixels
                    if eyebrow_thickness > 0.5:  # Very thick eyebrows
                        male_score += 1
                    elif eyebrow_thickness < 0.1:  # Very thin eyebrows
                        female_score += 1
                
                # Eye shape analysis
                eye_region_y1 = max(0, int(left_eye[1] - face_height * 0.05))
                eye_region_y2 = min(face_region.shape[0], int(left_eye[1] + face_height * 0.05))
                eye_region_x1 = max(0, int(left_eye[0] - face_width * 0.1))
                eye_region_x2 = min(face_region.shape[1], int(right_eye[0] + face_width * 0.1))
                
                if (eye_region_y1 < eye_region_y2 and eye_region_x1 < eye_region_x2):
                    eye_region = face_region[eye_region_y1:eye_region_y2, eye_region_x1:eye_region_x2]
                    
                    if eye_region.size > 0:
                        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                        
                        # Analyze eye shape (simplified)
                        # Females often have more almond-shaped eyes
                        eye_edges = cv2.Canny(gray_eye, 30, 100)
                        eye_edge_density = np.sum(eye_edges > 0) / (eye_region.shape[0] * eye_region.shape[1])
                        
                        if eye_edge_density > 0.1:  # More defined eye shape
                            female_score += 1
            except:
                pass
        
        # 6. Cheekbone Analysis
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            try:
                left_eye = face.kps[0]
                right_eye = face.kps[1]
                nose = face.kps[2]
                
                # Analyze cheekbone prominence
                eye_center = (left_eye + right_eye) / 2
                cheek_width = np.linalg.norm(right_eye - left_eye)
                face_center_to_nose = np.linalg.norm(nose - eye_center)
                
                if cheek_width > 0:
                    cheek_ratio = face_center_to_nose / cheek_width
                    # More prominent cheekbones (higher ratio) are more feminine
                    if cheek_ratio > 0.4:
                        female_score += 1
                    elif cheek_ratio < 0.3:
                        male_score += 1
            except:
                pass
        
        # 7. Lip and Mouth Analysis
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            try:
                left_mouth = face.kps[3]
                right_mouth = face.kps[4]
                
                # Analyze lip region
                lip_region_y1 = max(0, int(left_mouth[1] - face_height * 0.05))
                lip_region_y2 = min(face_region.shape[0], int(left_mouth[1] + face_height * 0.05))
                lip_region_x1 = max(0, int(left_mouth[0] - face_width * 0.1))
                lip_region_x2 = min(face_region.shape[1], int(right_mouth[0] + face_width * 0.1))
                
                if (lip_region_y1 < lip_region_y2 and lip_region_x1 < lip_region_x2):
                    lip_region = face_region[lip_region_y1:lip_region_y2, lip_region_x1:lip_region_x2]
                    
                    if lip_region.size > 0:
                        # Convert to HSV for better lip detection
                        hsv_lip = cv2.cvtColor(lip_region, cv2.COLOR_BGR2HSV)
                        
                        # Look for lip color (reddish tones)
                        lip_pixels = np.sum((hsv_lip[:,:,0] < 10) | (hsv_lip[:,:,0] > 170)) & (hsv_lip[:,:,1] > 50) & (hsv_lip[:,:,2] > 50)
                        total_lip_pixels = lip_region.shape[0] * lip_region.shape[1]
                        
                        if total_lip_pixels > 0:
                            lip_ratio = lip_pixels / total_lip_pixels
                            
                            # Females often have more defined lips
                            if lip_ratio > 0.4:  # Very well-defined lips
                                female_score += 1
                            elif lip_ratio < 0.05:  # Very less defined lips
                                male_score += 1
            except:
                pass
        
        # 8. Age-based adjustment (younger faces are harder to classify)
        if hasattr(face, 'age') and face.age is not None:
            age = face.age
            if age < 16:  # Teenagers are harder to classify
                # Reduce confidence for young faces
                male_score *= 0.7
                female_score *= 0.7
        
        # Simple and effective decision logic
        logger.info(f"🔍 Gender scores - Male: {male_score}, Female: {female_score}")
        
        # Very balanced decision logic
        if male_score > female_score + 2:  # Need clear male advantage
            return 'Male'
        elif female_score > male_score:  # Any female advantage wins
            return 'Female'
        else:
            # Tie-breaker: favor female for balance
            return 'Female'
            
    except Exception as e:
        logger.error(f"❌ Advanced gender classification error: {e}")
        return 'Unknown'

def classify_gender_simple(face, image):
    """Fallback simple gender classification"""
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        face_height = y2 - y1
        face_width = x2 - x1
        aspect_ratio = face_width / face_height if face_height > 0 else 1
        
        # Simple fallback
        if aspect_ratio > 0.82:
            return 'Male'
        elif aspect_ratio < 0.75:
            return 'Female'
        else:
            return 'Male' if aspect_ratio > 0.78 else 'Female'
            
    except Exception as e:
        logger.error(f"❌ Simple gender classification error: {e}")
        return 'Unknown'

def draw_rectangles(image, results):
    """Draw bounding boxes and labels on image"""
    image_with_boxes = image.copy()
    
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        gender = result['gender']
        confidence = result['confidence']
        
        # Choose color based on gender
        if gender == 'Female':
            color = (0, 255, 0)  # Green for female
        else:
            color = (255, 0, 0)  # Blue for male
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{gender}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_with_boxes

@app.route('/')
def index():
    """Main page"""
    return render_template('insightface.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame sent from browser"""
    global detection_results, frame_count
    
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
        # Keep a reference for classifiers
        global last_frame_image
        last_frame_image = image
        
        # Resize image for better speed (max 512px width for faster processing)
        h, w = image.shape[:2]
        if w > 512:
            scale = 512 / w
            new_w = 512
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Try multiple image orientations for better side face detection
        original_image = image.copy()
        rotated_images = [
            (original_image, 0),  # Original
            (cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE), 90),  # 90° clockwise
            (cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE), -90),  # 90° counter-clockwise
            (cv2.rotate(original_image, cv2.ROTATE_180), 180)  # 180°
        ]
        
        # Try face detection on multiple orientations for better side face detection
        all_results = []
        
        # Check if we have cached tracked faces to use
        if len(tracked_faces) > 0 and (frame_count - last_full_detection) < DETECTION_INTERVAL:
            logger.info(f"🔄 Using cached tracked faces (frame {frame_count})")
            # Use cached results from tracked faces
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
            for img, rotation in rotated_images:
                try:
                    # Route by detector
                    if current_detector == 'insightface':
                        results = detect_faces_insightface(img)
                    elif current_detector == 'yunet':
                        yunet_results = detect_faces_yunet(img)
                        # Convert YuNet mock results to tracked pipeline
                        converted = []
                        for r in yunet_results:
                            if 'mock_face' in r:
                                converted.append(r['mock_face'])
                        # Update tracking once per rotation (first non-empty)
                        if converted:
                            tracking_results = update_tracked_faces(converted, time.time())
                            # Build standard results list
                            tmp = []
                            for tr in tracking_results:
                                face = tr['face']
                                bbox = face.bbox.astype(int)
                                tmp.append({
                                    'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                    'gender': tr['gender'],
                                    'age': tr['age'],
                                    'confidence': float(tr['confidence']),
                                    'det_score': float(getattr(face, 'det_score', 0.8)),
                                    'face_id': tr['face_id'],
                                    'tracked': tr['tracked']
                                })
                            results = tmp
                        else:
                            results = []
                    else:
                        # Default to insightface path
                        results = detect_faces_insightface(img)
                    if results:
                        # Adjust bounding boxes for rotation
                        for result in results:
                            if rotation != 0:
                                # Rotate bounding box back to original orientation
                                bbox = result['bbox']
                                x1, y1, x2, y2 = bbox
                            
                                if rotation == 90:
                                    # 90° clockwise: (x,y) -> (y, h-x)
                                    h_rot, w_rot = img.shape[:2]
                                    new_x1, new_y1 = y1, h_rot - x2
                                    new_x2, new_y2 = y2, h_rot - x1
                                elif rotation == -90:
                                    # 90° counter-clockwise: (x,y) -> (w-y, x)
                                    h_rot, w_rot = img.shape[:2]
                                    new_x1, new_y1 = w_rot - y2, x1
                                    new_x2, new_y2 = w_rot - y1, x2
                                elif rotation == 180:
                                    # 180°: (x,y) -> (w-x, h-y)
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
        
        # Remove duplicate faces from different rotations
        results = remove_duplicate_faces_from_results(all_results)
        detection_results = results
        
        # If no faces detected, return early
        if len(results) == 0:
            logger.info("❌ No faces detected in this frame")
            # Encode image without rectangles
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 30])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                "image": f"data:image/jpeg;base64,{image_base64}",
                "faces": [],
                "fps": 0
            })
        
        # Draw rectangles on image for visualization (simplified)
        try:
            # Create a copy to avoid modifying original
            image_with_boxes = image.copy()
            
            # Draw simple rectangles
            for result in results:
                bbox = result['bbox']
                x1, y1, x2, y2 = bbox
                gender = result['gender']
                confidence = result['confidence']
                
                # Choose color based on gender
                color = (0, 255, 0) if gender == 'Female' else (255, 0, 0)  # Green for Female, Blue for Male
                
                # Draw rectangle
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{gender} ({confidence:.2f})"
                cv2.putText(image_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Encode image with rectangles (balanced quality for better accuracy)
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
            # Return results without image processing to prevent crashes
            # Prepare performance metrics for error case
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
                "image_with_boxes": None,
                **performance_metrics
            })
        finally:
            # Performance monitoring
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Log performance metrics every 50 frames
            if frame_count % 50 == 0:
                tracked_count = len(tracked_faces)
                logger.info(f"📊 Performance - Frame: {frame_count}, Processing time: {processing_time:.3f}s, FPS: {fps:.1f}, Tracked faces: {tracked_count}")
            
            # Clean up memory for 24/7 operation
            if 'image' in locals():
                del image
            if 'image_with_boxes' in locals():
                del image_with_boxes
            if 'image_bytes' in locals():
                del image_bytes
            
            # Force garbage collection every 50 frames for Pi 5 (more frequent)
            if frame_count % 50 == 0:
                import gc
                gc.collect()
                logger.info("🧹 Memory cleanup performed")
        
    except Exception as e:
        logger.error(f"❌ Frame processing error: {e}")
        return jsonify({"error": str(e)})

@app.route('/get_results')
def get_results():
    """Get current detection results"""
    return jsonify(detection_results)

if __name__ == '__main__':
    # Initialize InsightFace
    initialize_insightface()
    
    # Start Flask app
    logger.info("🚀 Starting InsightFace web server...")
    logger.info("🌐 Access: http://172.22.196.192:5000")
    logger.info("📱 Phone Access: http://10.54.91.236:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
