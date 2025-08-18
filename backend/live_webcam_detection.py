"""
Live Webcam Gender Detection for Bus Safety System
Integrates with backend services for real-time detection
"""

import cv2
import numpy as np
import time
import base64
from datetime import datetime
from gender_detection_service import GenderDetectionService
from appwrite_service import AppwriteService
from config import Config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveWebcamDetection:
    def __init__(self, bus_id="LIVE_BUS_001", route_number="118"):
        self.bus_id = bus_id
        self.route_number = route_number
        self.gender_service = GenderDetectionService()
        self.appwrite_service = AppwriteService()
        self.frame_interval = 4  # Grab frame every 4 seconds
        self.last_frame_time = 0
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Cannot open webcam!")
        
        logger.info(f"🎥 Webcam initialized for bus {bus_id} on route {route_number}")
        
    def process_frame(self, frame):
        """Process a single frame for gender detection"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Process with gender detection service
            face_count, gender_counts, confidence_scores = self.gender_service.process_image(img_base64)
            
            return face_count, gender_counts, confidence_scores
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return 0, {'male': 0, 'female': 0, 'unknown': 0}, {}
    
    def save_to_database(self, gender_counts, confidence_scores):
        """Save detection results to Appwrite database"""
        try:
            # Calculate average confidence
            if confidence_scores:
                avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            else:
                avg_confidence = 0.0
            
            # Create gender count object
            gender_count = self.gender_service.create_gender_count(
                self.bus_id, gender_counts, avg_confidence
            )
            
            # Save to database
            success = self.appwrite_service.save_gender_count(gender_count)
            
            if success:
                logger.info(f"💾 Saved to database: {gender_counts}")
            else:
                logger.warning("⚠️ Failed to save to database")
                
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def draw_results(self, frame, face_count, gender_counts, confidence_scores, processing_time):
        """Draw detection results on frame"""
        # Create overlay
        overlay = frame.copy()
        
        # Header
        cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Bus {self.bus_id} - Route {self.route_number}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Live Detection - {datetime.now().strftime('%H:%M:%S')}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Results panel
        panel_y = 80
        cv2.rectangle(overlay, (10, panel_y), (300, panel_y + 120), (0, 0, 0), -1)
        
        # Face count
        cv2.putText(frame, f"Faces: {face_count}", (20, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Gender counts with colors
        y_offset = 50
        for gender, count in gender_counts.items():
            if count > 0:
                color = (0, 255, 0) if gender == 'female' else (0, 0, 255) if gender == 'male' else (128, 128, 128)
                cv2.putText(frame, f"{gender.title()}: {count}", (20, panel_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        # Processing info
        cv2.putText(frame, f"Frame: {processing_time:.2f}s", (20, panel_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Frame interval indicator
        time_since_last = time.time() - self.last_frame_time
        if time_since_last < self.frame_interval:
            remaining = self.frame_interval - time_since_last
            cv2.putText(frame, f"Next frame in: {remaining:.1f}s", (320, panel_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Processing frame...", (320, panel_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Footer
        cv2.rectangle(overlay, (0, 420), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame", (10, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Frame interval: 4s | Memory optimized", (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main detection loop"""
        logger.info("🚀 Starting live webcam detection...")
        logger.info("💡 Press 'q' to quit, 's' to save current frame")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("❌ Failed to grab frame")
                    break
                
                current_time = time.time()
                
                # Process frame every 4 seconds to save memory
                if current_time - self.last_frame_time >= self.frame_interval:
                    logger.info("🔍 Processing new frame...")
                    
                    start_time = time.time()
                    face_count, gender_counts, confidence_scores = self.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    # Save to database
                    self.save_to_database(gender_counts, confidence_scores)
                    
                    self.last_frame_time = current_time
                    logger.info(f"✅ Processed {face_count} faces in {processing_time:.2f}s")
                
                # Draw results on frame
                self.draw_results(frame, face_count, gender_counts, confidence_scores, processing_time)
                
                # Display frame
                cv2.imshow('Live Gender Detection - Bus Safety System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"live_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"💾 Saved frame as {filename}")
                
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("✅ Cleanup completed")

def main():
    """Main function"""
    print("🚌 Live Webcam Gender Detection for Bus Safety System")
    print("=" * 60)
    
    try:
        # Initialize detection system
        detector = LiveWebcamDetection()
        
        # Start detection
        detector.run()
        
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        logger.error(f"Startup failed: {e}")

if __name__ == "__main__":
    main()
