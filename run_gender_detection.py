#!/usr/bin/env python3
"""
Simple Gender Detection Inference Script
Run your trained models on laptop/desktop
No GPU training needed - just inference!
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time
import argparse

class GenderDetector:
    def __init__(self, model_path='gender_detection_modern.keras'):
        """Initialize the gender detector with a trained model"""
        self.model_path = Path(model_path)
        self.model = None
        self.class_names = ['Man', 'Woman']
        self.img_size = 224
        
        # Load the model
        self.load_model()
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def load_model(self):
        """Load the trained model"""
        print(f"🔄 Loading model from: {self.model_path}")
        
        try:
            # Try to load the model
            self.model = tf.keras.models.load_model(str(self.model_path))
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model summary:")
            self.model.summary()
            
        except Exception as e:
            print(f"❌ Failed to load {self.model_path}: {e}")
            
            # Try alternative models
            alternative_models = [
                'best_gender_model.keras',
                'gender_detection.model'
            ]
            
            for alt_model in alternative_models:
                if Path(alt_model).exists():
                    print(f"🔄 Trying alternative model: {alt_model}")
                    try:
                        self.model = tf.keras.models.load_model(alt_model)
                        print(f"✅ Alternative model loaded: {alt_model}")
                        break
                    except Exception as e2:
                        print(f"❌ Failed to load {alt_model}: {e2}")
                        continue
            
            if self.model is None:
                raise ValueError("No working model found!")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Resize to model input size
        image_resized = cv2.resize(image_rgb, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_gender(self, image):
        """Predict gender from image"""
        if self.model is None:
            return None, 0.0
            
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return faces
    
    def run_webcam(self):
        """Run real-time gender detection on webcam"""
        print("🎥 Starting webcam...")
        print("📱 Press 'q' to quit, 's' to save image")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Predict gender
                gender_class, confidence = self.predict_gender(face_img)
                
                if gender_class is not None:
                    gender = self.class_names[gender_class]
                    
                    # Draw rectangle around face
                    color = (0, 255, 0) if gender == 'Man' else (255, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw gender label
                    label = f"{gender}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, 
                                (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), 
                                color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - start_time)
                start_time = current_time
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Gender Detection - Press q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"gender_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Image saved as: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Webcam session ended")
    
    def detect_from_image(self, image_path):
        """Detect gender from a single image file"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        print(f"🖼️  Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("❌ No faces detected in the image")
            return
        
        print(f"👥 Found {len(faces)} face(s)")
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Predict gender
            gender_class, confidence = self.predict_gender(face_img)
            
            if gender_class is not None:
                gender = self.class_names[gender_class]
                
                print(f"👤 Face {i+1}: {gender} (Confidence: {confidence:.2f})")
                
                # Draw rectangle and label
                color = (0, 255, 0) if gender == 'Man' else (255, 0, 255)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                
                label = f"{gender}: {confidence:.2f}"
                cv2.putText(image, label, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save annotated image
        output_path = f"annotated_{image_path.stem}.jpg"
        cv2.imwrite(output_path, image)
        print(f"💾 Annotated image saved as: {output_path}")
        
        # Show image
        cv2.imshow('Gender Detection Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def batch_detect(self, folder_path):
        """Detect gender from all images in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ Folder not found: {folder_path}")
            return
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"❌ No image files found in: {folder_path}")
            return
        
        print(f"🖼️  Found {len(image_files)} images to process")
        
        results = []
        
        for i, image_file in enumerate(image_files):
            print(f"\n📊 Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    print(f"❌ Failed to load: {image_file.name}")
                    continue
                
                # Detect faces
                faces = self.detect_faces(image)
                
                if len(faces) == 0:
                    print(f"❌ No faces in: {image_file.name}")
                    continue
                
                # Process each face
                for j, (x, y, w, h) in enumerate(faces):
                    face_img = image[y:y+h, x:x+w]
                    gender_class, confidence = self.predict_gender(face_img)
                    
                    if gender_class is not None:
                        gender = self.class_names[gender_class]
                        results.append({
                            'image': image_file.name,
                            'face': j+1,
                            'gender': gender,
                            'confidence': confidence
                        })
                        
                        print(f"  👤 Face {j+1}: {gender} ({confidence:.2f})")
                
            except Exception as e:
                print(f"❌ Error processing {image_file.name}: {e}")
        
        # Print summary
        print(f"\n📊 Batch processing complete!")
        print(f"✅ Processed {len(image_files)} images")
        print(f"👥 Found {len(results)} faces total")
        
        if results:
            men = sum(1 for r in results if r['gender'] == 'Man')
            women = sum(1 for r in results if r['gender'] == 'Woman')
            print(f"👨 Men: {men}")
            print(f"👩 Women: {women}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Gender Detection Inference')
    parser.add_argument('--mode', choices=['webcam', 'image', 'batch'], 
                       default='webcam', help='Detection mode')
    parser.add_argument('--model', default='gender_detection_modern.keras',
                       help='Path to trained model')
    parser.add_argument('--input', help='Input image or folder path')
    
    args = parser.parse_args()
    
    print("🚀 Gender Detection Inference Script")
    print("=" * 50)
    print(f"🎯 Mode: {args.mode}")
    print(f"🏗️  Model: {args.model}")
    
    try:
        # Create detector
        detector = GenderDetector(args.model)
        
        # Run based on mode
        if args.mode == 'webcam':
            detector.run_webcam()
        elif args.mode == 'image':
            if not args.input:
                print("❌ Please provide input image path with --input")
                return
            detector.detect_from_image(args.input)
        elif args.mode == 'batch':
            if not args.input:
                print("❌ Please provide input folder path with --input")
                return
            detector.batch_detect(args.input)
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()








