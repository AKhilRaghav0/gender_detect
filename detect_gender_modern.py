#!/usr/bin/env python3
"""
Modern Gender Detection Inference Script
Using ResNet50 model with advanced face detection
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from pathlib import Path
import argparse

class ModernGenderDetector:
    def __init__(self, 
                 model_path='gender_detection_modern.keras',
                 config_path='model_config.json',
                 use_gpu=True):
        
        self.model_path = model_path
        self.config_path = config_path
        self.use_gpu = use_gpu
        
        # Setup GPU first
        self.setup_gpu()
        
        # Load configuration
        self.load_config()
        
        # Load model
        self.load_model()
        
        # Initialize face detector (using OpenCV's DNN)
        self.setup_face_detector()
        
        print("🚀 Modern Gender Detector initialized")
        print(f"📱 Model: {self.model_path}")
        print(f"🖼️  Input size: {self.img_size}x{self.img_size}")
        print(f"👥 Classes: {self.class_names}")
        print(f"🎮 GPU: {'Enabled' if self.gpu_available else 'Disabled'}")
    
    def setup_gpu(self):
        """Setup GPU for inference"""
        self.gpu_available = False
        
        if not self.use_gpu:
            print("🖥️  GPU disabled by user")
            return
        
        try:
            import tensorflow as tf
            
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Configure GPU memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                self.gpu_available = True
                print(f"✅ GPU configured: {len(gpus)} GPU(s) available for inference")
                
                # Get GPU details
                for i, gpu in enumerate(gpus):
                    try:
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                        print(f"  GPU {i}: {gpu_name}")
                        
                        # Optimize for specific GPU types
                        if 'RTX 3050' in gpu_name:
                            print("  🚀 RTX 3050 detected - Optimized for fast inference")
                        elif 'GTX 12' in gpu_name:
                            print("  🚀 GTX 1200 series detected - Good inference performance")
                        elif 'RTX' in gpu_name:
                            print("  🚀 RTX series detected - Excellent inference performance")
                            
                    except Exception as e:
                        print(f"  Could not get GPU {i} details: {e}")
            else:
                print("🖥️  No GPU detected, using CPU for inference")
                
        except Exception as e:
            print(f"❌ GPU setup error: {e}")
            print("🖥️  Falling back to CPU")
        
    def load_config(self):
        """Load model configuration"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.img_size = config.get('img_size', 224)
            self.class_names = config.get('class_names', ['man', 'woman'])
            self.num_classes = config.get('num_classes', 2)
        else:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("Using default configuration...")
            self.img_size = 224
            self.class_names = ['man', 'woman']
            self.num_classes = 2
    
    def load_model(self):
        """Load the trained model"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please train the model first by running: python train_modern.py")
            raise
    
    def setup_face_detector(self):
        """Setup OpenCV DNN face detector"""
        try:
            # Download models if they don't exist
            self.download_face_detection_models()
            
            # Load the DNN model
            self.face_net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            print("✅ Face detector loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading face detector: {e}")
            print("Falling back to Haar Cascade...")
            self.setup_haar_cascade()
    
    def download_face_detection_models(self):
        """Download OpenCV DNN face detection models if not present"""
        import urllib.request
        
        model_files = {
            'opencv_face_detector_uint8.pb': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
            'opencv_face_detector.pbtxt': 'https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt'
        }
        
        for filename, url in model_files.items():
            if not Path(filename).exists():
                print(f"📥 Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filename)
                    print(f"✅ Downloaded {filename}")
                except Exception as e:
                    print(f"❌ Failed to download {filename}: {e}")
                    raise
    
    def setup_haar_cascade(self):
        """Fallback to Haar Cascade face detector"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.use_dnn = False
            print("✅ Haar Cascade face detector loaded")
        except Exception as e:
            print(f"❌ Error loading Haar Cascade: {e}")
            raise
    
    def detect_faces_dnn(self, frame):
        """Detect faces using OpenCV DNN"""
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), [104, 117, 123]
        )
        
        # Set input to the network
        self.face_net.setInput(blob)
        
        # Run forward pass
        detections = self.face_net.forward()
        
        faces = []
        confidences = []
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(float(confidence))
        
        return faces, confidences
    
    def detect_faces_haar(self, frame):
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        confidences = [1.0] * len(faces)  # Haar doesn't provide confidence
        return faces.tolist(), confidences
    
    def detect_faces(self, frame):
        """Detect faces using the available detector"""
        if hasattr(self, 'face_net'):
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def preprocess_face(self, face_img):
        """Preprocess face image for gender prediction"""
        # Resize to model input size
        face_resized = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_gender(self, face_img):
        """Predict gender for a face image"""
        # Preprocess face
        face_processed = self.preprocess_face(face_img)
        
        # Make prediction
        predictions = self.model.predict(face_processed, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        gender = self.class_names[predicted_class]
        
        return gender, confidence
    
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        # Detect faces
        faces, face_confidences = self.detect_faces(frame)
        
        results = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            if face_img.size > 0:
                # Predict gender
                gender, gender_confidence = self.predict_gender(face_img)
                
                results.append({
                    'bbox': (x, y, w, h),
                    'gender': gender,
                    'gender_confidence': gender_confidence,
                    'face_confidence': face_confidences[i]
                })
        
        return results
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        for result in results:
            x, y, w, h = result['bbox']
            gender = result['gender']
            gender_conf = result['gender_confidence']
            
            # Choose color based on gender
            color = (255, 0, 0) if gender == 'man' else (0, 255, 255)  # Blue for man, Yellow for woman
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label = f"{gender}: {gender_conf:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """Run real-time gender detection on webcam"""
        print(f"🎥 Starting webcam detection (Camera ID: {camera_id})")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw results
                frame_with_results = self.draw_results(frame, results)
                
                # Calculate and display FPS
                fps_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    start_time = time.time()
                    
                    # Display FPS on frame
                    cv2.putText(
                        frame_with_results,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                # Display frame
                cv2.imshow('Modern Gender Detection', frame_with_results)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_results)
                    print(f"📸 Screenshot saved: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🎥 Webcam session ended")
    
    def process_image(self, image_path, output_path=None):
        """Process a single image file"""
        print(f"🖼️  Processing image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Process image
        results = self.process_frame(frame)
        
        # Draw results
        frame_with_results = self.draw_results(frame, results)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, frame_with_results)
            print(f"💾 Result saved: {output_path}")
        else:
            cv2.imshow('Gender Detection Result', frame_with_results)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print results
        print(f"📊 Detected {len(results)} face(s):")
        for i, result in enumerate(results):
            print(f"  Face {i+1}: {result['gender']} ({result['gender_confidence']:.3f})")
        
        return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Modern Gender Detection')
    parser.add_argument('--mode', choices=['webcam', 'image'], default='webcam',
                       help='Detection mode (default: webcam)')
    parser.add_argument('--image', type=str,
                       help='Path to input image (for image mode)')
    parser.add_argument('--output', type=str,
                       help='Path to output image (for image mode)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam mode (default: 0)')
    parser.add_argument('--model', type=str, default='gender_detection_modern.keras',
                       help='Path to model file')
    parser.add_argument('--config', type=str, default='model_config.json',
                       help='Path to config file')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Handle GPU arguments
    use_gpu = args.gpu and not args.no_gpu
    
    print("=" * 60)
    print("🚀 Modern Gender Detection")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = ModernGenderDetector(
            model_path=args.model,
            config_path=args.config,
            use_gpu=use_gpu
        )
        
        if args.mode == 'webcam':
            detector.run_webcam(camera_id=args.camera)
        
        elif args.mode == 'image':
            if not args.image:
                print("❌ Please specify --image path for image mode")
                return
            
            detector.process_image(args.image, args.output)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
