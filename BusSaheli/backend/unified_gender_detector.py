#!/usr/bin/env python3
"""
Unified Gender Detection System
Consolidates all detection algorithms into a single, maintainable interface
"""

import cv2
import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionAlgorithm(Enum):
    """Available detection algorithms"""
    HAAR_CASCADE = "haar_cascade"
    YUNET = "yunet"
    SCRFD = "scrfd"
    INSIGHTFACE = "insightface"
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    BALANCED = "balanced"
    PREMIUM = "premium"
    ULTRA_SENSITIVE = "ultra_sensitive"
    POLISHED = "polished"
    SIMPLE_LOGIC = "simple_logic"

@dataclass
class DetectionResult:
    """Standardized detection result"""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    gender: str  # "Male" or "Female"
    confidence: float  # 0.0 to 1.0
    face_id: str
    timestamp: float
    algorithm_used: str
    features: Optional[Dict[str, float]] = None

@dataclass
class DetectionConfig:
    """Configuration for detection algorithms"""
    algorithm: DetectionAlgorithm
    confidence_threshold: float = 0.5
    min_face_size: Tuple[int, int] = (50, 50)
    max_faces: int = 10
    enable_tracking: bool = True
    memory_cleanup_interval: int = 50

class BaseDetector(ABC):
    """Abstract base class for all detection algorithms"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_count = 0
        self.last_cleanup = time.time()
        self.face_tracker = {}  # For face tracking
        
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces and return standardized results"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        self.frame_count += 1
        
        # Time-based cleanup (every 5 seconds)
        if time.time() - self.last_cleanup > 5:
            return True
        
        # Frame-based cleanup
        if self.frame_count % self.config.memory_cleanup_interval == 0:
            return True
        
        return False
    
    def auto_cleanup(self):
        """Perform automatic cleanup if needed"""
        if self.should_cleanup():
            self.cleanup()
            self.last_cleanup = time.time()
            logger.debug(f"🧹 Auto cleanup performed for {self.config.algorithm.value}")

class HaarCascadeDetector(BaseDetector):
    """Haar Cascade-based detection (most reliable fallback)"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        logger.info("✅ Haar Cascade detector initialized")
    
    def detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            1.1, 
            5, 
            minSize=self.config.min_face_size
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Simple gender detection based on face proportions
            gender, confidence = self._detect_gender_simple(image[y:y+h, x:x+w])
            
            result = DetectionResult(
                bbox=(x, y, w, h),
                gender=gender,
                confidence=confidence,
                face_id=f"haar_{self.frame_count}_{i}",
                timestamp=time.time(),
                algorithm_used="haar_cascade"
            )
            results.append(result)
        
        return results
    
    def _detect_gender_simple(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Simple gender detection based on face proportions"""
        h, w = face_roi.shape[:2]
        
        # Basic facial feature analysis
        face_width_ratio = w / h
        eye_spacing_ratio = 0.5  # Placeholder - would need eye detection
        
        # Simple heuristic (can be improved)
        if face_width_ratio > 0.8:
            return "Male", 0.6
        else:
            return "Female", 0.6
    
    def cleanup(self):
        """Clean up resources"""
        logger.debug("🧹 Haar Cascade detector cleanup")

class InsightFaceDetector(BaseDetector):
    """InsightFace-based detection (highest accuracy)"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # Initialize InsightFace
            self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info("✅ InsightFace detector initialized")
            self.available = True
            
        except ImportError:
            logger.warning("⚠️ InsightFace not available, falling back to Haar Cascade")
            self.available = False
            # Fallback to Haar Cascade
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces using InsightFace"""
        if not self.available:
            return self._fallback_detection(image)
        
        try:
            # Use InsightFace for detection
            faces = self.face_app.get(image)
            
            results = []
            for i, face in enumerate(faces):
                # Get bounding box
                bbox = face.bbox.astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                # Get gender prediction
                gender = "Female" if face.sex == 0 else "Male"
                confidence = face.sex_score
                
                result = DetectionResult(
                    bbox=(x, y, w, h),
                    gender=gender,
                    confidence=confidence,
                    face_id=f"insightface_{self.frame_count}_{i}",
                    timestamp=time.time(),
                    algorithm_used="insightface"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ InsightFace detection failed: {e}")
            return self._fallback_detection(image)
    
    def _fallback_detection(self, image: np.ndarray) -> List[DetectionResult]:
        """Fallback to Haar Cascade if InsightFace fails"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Simple gender detection
            gender, confidence = self._detect_gender_simple(image[y:y+h, x:x+w])
            
            result = DetectionResult(
                bbox=(x, y, w, h),
                gender=gender,
                confidence=confidence,
                face_id=f"haar_fallback_{self.frame_count}_{i}",
                timestamp=time.time(),
                algorithm_used="haar_cascade_fallback"
            )
            results.append(result)
        
        return results
    
    def _detect_gender_simple(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Simple gender detection fallback"""
        h, w = face_roi.shape[:2]
        face_width_ratio = w / h
        
        if face_width_ratio > 0.8:
            return "Male", 0.6
        else:
            return "Female", 0.6
    
    def cleanup(self):
        """Clean up InsightFace resources"""
        if hasattr(self, 'face_app'):
            del self.face_app
        logger.debug("🧹 InsightFace detector cleanup")

class ProfessionalDetector(BaseDetector):
    """Professional-grade detection with advanced features"""
    
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Professional gender detection parameters
        self.gender_weights = {
            'face_width_ratio': 0.25,
            'eye_spacing_ratio': 0.20,
            'jaw_line_ratio': 0.25,
            'cheekbone_ratio': 0.15,
            'forehead_ratio': 0.15
        }
        
        self.female_threshold = 0.45
        logger.info("✅ Professional detector initialized")
    
    def detect_faces(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect faces with professional-grade analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            1.06, 
            3, 
            minSize=self.config.min_face_size
        )
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y+h, x:x+w]
            gender, confidence, features = self._analyze_face_professional(face_roi)
            
            result = DetectionResult(
                bbox=(x, y, w, h),
                gender=gender,
                confidence=confidence,
                face_id=f"professional_{self.frame_count}_{i}",
                timestamp=time.time(),
                algorithm_used="professional",
                features=features
            )
            results.append(result)
        
        return results
    
    def _analyze_face_professional(self, face_roi: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Professional face analysis"""
        h, w = face_roi.shape[:2]
        
        # Calculate facial features
        features = {
            'face_width_ratio': w / h,
            'eye_spacing_ratio': 0.5,  # Placeholder
            'jaw_line_ratio': 0.5,    # Placeholder
            'cheekbone_ratio': 0.5,   # Placeholder
            'forehead_ratio': 0.5     # Placeholder
        }
        
        # Calculate gender score
        gender_score = sum(
            features[feature] * weight 
            for feature, weight in self.gender_weights.items()
        )
        
        if gender_score < self.female_threshold:
            return "Female", min(gender_score + 0.3, 1.0), features
        else:
            return "Male", min(gender_score + 0.3, 1.0), features
    
    def cleanup(self):
        """Clean up resources"""
        logger.debug("🧹 Professional detector cleanup")

class UnifiedGenderDetector:
    """Unified interface for all gender detection algorithms"""
    
    def __init__(self, algorithm: DetectionAlgorithm = DetectionAlgorithm.HAAR_CASCADE):
        self.algorithm = algorithm
        self.config = DetectionConfig(algorithm=algorithm)
        self.detector = self._create_detector()
        self.detection_history = []
        self.performance_stats = {
            'total_detections': 0,
            'average_confidence': 0.0,
            'processing_times': []
        }
        
        logger.info(f"🚀 Unified Gender Detector initialized with {algorithm.value}")
    
    def _create_detector(self) -> BaseDetector:
        """Create the appropriate detector based on algorithm"""
        if self.algorithm == DetectionAlgorithm.HAAR_CASCADE:
            return HaarCascadeDetector(self.config)
        elif self.algorithm == DetectionAlgorithm.INSIGHTFACE:
            return InsightFaceDetector(self.config)
        elif self.algorithm == DetectionAlgorithm.PROFESSIONAL:
            return ProfessionalDetector(self.config)
        else:
            # Fallback to Haar Cascade for unsupported algorithms
            logger.warning(f"Algorithm {self.algorithm.value} not implemented, using Haar Cascade")
            return HaarCascadeDetector(self.config)
    
    def detect_gender(self, image: np.ndarray) -> List[DetectionResult]:
        """Main detection method"""
        start_time = time.time()
        
        try:
            # Perform detection
            results = self.detector.detect_faces(image)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['processing_times'].append(processing_time)
            self.performance_stats['total_detections'] += len(results)
            
            if results:
                avg_confidence = sum(r.confidence for r in results) / len(results)
                self.performance_stats['average_confidence'] = avg_confidence
            
            # Store detection history
            self.detection_history.extend(results)
            
            # Auto cleanup
            self.detector.auto_cleanup()
            
            logger.debug(f"🔍 Detected {len(results)} faces in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if self.performance_stats['processing_times']:
            avg_time = sum(self.performance_stats['processing_times']) / len(self.performance_stats['processing_times'])
            self.performance_stats['average_processing_time'] = avg_time
        
        return self.performance_stats.copy()
    
    def cleanup(self):
        """Clean up all resources"""
        if self.detector:
            self.detector.cleanup()
        
        # Clear history to prevent memory leaks
        self.detection_history.clear()
        
        # Force garbage collection
        gc.collect()
        logger.info("🧹 Unified detector cleanup completed")
    
    def switch_algorithm(self, new_algorithm: DetectionAlgorithm):
        """Switch to a different detection algorithm"""
        logger.info(f"🔄 Switching from {self.algorithm.value} to {new_algorithm.value}")
        
        # Cleanup current detector
        if self.detector:
            self.detector.cleanup()
        
        # Switch algorithm
        self.algorithm = new_algorithm
        self.config.algorithm = new_algorithm
        self.detector = self._create_detector()
        
        logger.info(f"✅ Switched to {new_algorithm.value}")

# Factory function for easy creation
def create_gender_detector(algorithm: str = "haar_cascade") -> UnifiedGenderDetector:
    """Factory function to create a gender detector"""
    try:
        algo_enum = DetectionAlgorithm(algorithm)
        return UnifiedGenderDetector(algo_enum)
    except ValueError:
        logger.warning(f"Unknown algorithm {algorithm}, using Haar Cascade")
        return UnifiedGenderDetector(DetectionAlgorithm.HAAR_CASCADE)

# Example usage
if __name__ == "__main__":
    # Test the unified detector
    detector = create_gender_detector("professional")
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test detection
    results = detector.detect_gender(test_image)
    print(f"Detected {len(results)} faces")
    
    # Get performance stats
    stats = detector.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Cleanup
    detector.cleanup()
