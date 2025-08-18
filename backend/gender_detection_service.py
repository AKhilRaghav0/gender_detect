"""
Gender Detection Service for Bus Safety Gender Detection System
Integrates with existing polished detection system and provides real-time counting
"""

import cv2
import numpy as np
import base64
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from PIL import Image
import io

from config import Config
from models import GenderCount, BusSafetyMetrics, SafetyLevel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenderDetectionService:
    """Service class for gender detection and counting"""
    
    def __init__(self):
        """Initialize the gender detection service"""
        try:
            # Load OpenCV face detection cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Gender detection parameters (from polished_gender_detection.py)
            self.female_threshold = 0.40
            self.gender_weights = {
                'face_width': 0.25,
                'eye_spacing': 0.20,
                'jaw_strength': 0.25,
                'cheekbone_position': 0.15,
                'forehead_ratio': 0.15
            }
            
            logger.info("Gender detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize gender detection service: {e}")
            raise
    
    def process_image(self, image_data: str) -> Tuple[int, Dict[str, int], Dict[str, float]]:
        """
        Process base64 encoded image and return gender counts
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Tuple of (total_faces, gender_counts, confidence_scores)
        """
        try:
            start_time = time.time()
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Convert to OpenCV format (BGR)
            if len(image_np.shape) == 3:
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_cv = image_np
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.08, 
                minNeighbors=4, 
                minSize=(45, 45)
            )
            
            total_faces = len(faces)
            gender_counts = {'male': 0, 'female': 0, 'unknown': 0}
            confidence_scores = {'male': 0.0, 'female': 0.0, 'unknown': 0.0}
            
            if total_faces == 0:
                processing_time = time.time() - start_time
                logger.info(f"No faces detected in {processing_time:.3f}s")
                return total_faces, gender_counts, confidence_scores
            
            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Detect eyes in face region
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi, 
                    scaleFactor=1.1, 
                    minNeighbors=3, 
                    minSize=(15, 15)
                )
                
                # Analyze facial features
                gender_score = self._analyze_facial_features(face_roi, eyes, w, h)
                
                # Classify gender based on score
                if gender_score > self.female_threshold:
                    gender_counts['female'] += 1
                    confidence_scores['female'] = max(confidence_scores['female'], gender_score)
                else:
                    gender_counts['male'] += 1
                    confidence_scores['male'] = max(confidence_scores['male'], 1 - gender_score)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {total_faces} faces in {processing_time:.3f}s")
            
            return total_faces, gender_counts, confidence_scores
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return 0, {'male': 0, 'female': 0, 'unknown': 0}, {'male': 0.0, 'female': 0.0, 'unknown': 0.0}
    
    def _analyze_facial_features(self, face_roi: np.ndarray, eyes: np.ndarray, face_width: int, face_height: int) -> float:
        """
        Analyze facial features to determine gender probability
        
        Args:
            face_roi: Face region of interest (grayscale)
            eyes: Detected eyes in face region
            face_width: Width of detected face
            face_height: Height of detected face
            
        Returns:
            Gender score (0.0 = male, 1.0 = female)
        """
        try:
            # Initialize feature scores
            feature_scores = {}
            
            # 1. Face width analysis (females typically have narrower faces)
            face_ratio = face_width / face_height
            feature_scores['face_width'] = min(face_ratio / 0.8, 1.0)  # Normalize to 0-1
            
            # 2. Eye spacing analysis (females typically have wider eye spacing)
            if len(eyes) >= 2:
                eye_centers = []
                for (ex, ey, ew, eh) in eyes:
                    eye_center_x = ex + ew // 2
                    eye_centers.append(eye_center_x)
                
                if len(eye_centers) >= 2:
                    eye_spacing = abs(eye_centers[1] - eye_centers[0])
                    normalized_spacing = min(eye_spacing / face_width, 1.0)
                    feature_scores['eye_spacing'] = normalized_spacing
                else:
                    feature_scores['eye_spacing'] = 0.5
            else:
                feature_scores['eye_spacing'] = 0.5
            
            # 3. Jaw line strength analysis (males typically have stronger jaw lines)
            # Use edge detection to analyze jaw line
            edges = cv2.Canny(face_roi, 50, 150)
            jaw_region = edges[face_height//2:, :]
            jaw_strength = np.sum(jaw_region) / (jaw_region.shape[0] * jaw_region.shape[1])
            feature_scores['jaw_strength'] = min(jaw_strength / 100, 1.0)
            
            # 4. Cheekbone position analysis (females typically have higher cheekbones)
            cheekbone_region = face_roi[face_height//3:2*face_height//3, :]
            cheekbone_variance = np.var(cheekbone_region)
            feature_scores['cheekbone_position'] = min(cheekbone_variance / 1000, 1.0)
            
            # 5. Forehead ratio analysis (females typically have larger foreheads)
            forehead_region = face_roi[:face_height//3, :]
            forehead_ratio = np.mean(forehead_region) / np.mean(face_roi)
            feature_scores['forehead_ratio'] = min(forehead_ratio, 1.0)
            
            # Calculate weighted gender score
            total_score = 0.0
            total_weight = 0.0
            
            for feature, weight in self.gender_weights.items():
                if feature in feature_scores:
                    total_score += feature_scores[feature] * weight
                    total_weight += weight
            
            if total_weight > 0:
                gender_score = total_score / total_weight
            else:
                gender_score = 0.5
            
            return gender_score
            
        except Exception as e:
            logger.error(f"Error analyzing facial features: {e}")
            return 0.5  # Return neutral score on error
    
    def create_gender_count(self, bus_id: str, gender_counts: Dict[str, int], confidence: float) -> GenderCount:
        """
        Create a GenderCount object from detection results
        
        Args:
            bus_id: Bus identifier
            gender_counts: Dictionary with male/female/unknown counts
            confidence: Overall detection confidence
            
        Returns:
            GenderCount object
        """
        try:
            total_passengers = sum(gender_counts.values())
            
            return GenderCount(
                bus_id=bus_id,
                male_count=gender_counts.get('male', 0),
                female_count=gender_counts.get('female', 0),
                total_passengers=total_passengers,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Error creating gender count: {e}")
            return GenderCount(
                bus_id=bus_id,
                male_count=0,
                female_count=0,
                total_passengers=0,
                confidence_score=0.0
            )
    
    def calculate_safety_metrics(self, bus_id: str, gender_count: GenderCount, route_number: str) -> BusSafetyMetrics:
        """
        Calculate bus safety metrics based on gender count and route
        
        Args:
            bus_id: Bus identifier
            gender_count: Current gender count data
            route_number: Bus route number
            
        Returns:
            BusSafetyMetrics object
        """
        try:
            # Calculate female ratio
            if gender_count.total_passengers > 0:
                female_ratio = gender_count.female_count / gender_count.total_passengers
            else:
                female_ratio = 0.0
            
            # Calculate capacity utilization
            capacity_utilization = gender_count.total_passengers / Config.MAX_BUS_CAPACITY
            
            # Get route safety score
            route_info = Config.GURUGRAM_ROUTES.get(route_number, {})
            route_safety_score = route_info.get('safety_score', 0.7)
            
            # Calculate overall safety score
            safety_score = (
                female_ratio * Config.SAFETY_SCORE_WEIGHTS['female_ratio'] +
                (1 - capacity_utilization) * Config.SAFETY_SCORE_WEIGHTS['capacity_utilization'] +
                route_safety_score * Config.SAFETY_SCORE_WEIGHTS['route_safety']
            )
            
            # Determine safety level
            if safety_score >= 0.7:
                safety_level = SafetyLevel.SAFE
            elif safety_score >= 0.4:
                safety_level = SafetyLevel.MODERATE
            else:
                safety_level = SafetyLevel.UNSAFE
            
            # Generate recommendations
            recommendations = self._generate_safety_recommendations(
                female_ratio, capacity_utilization, route_safety_score, route_number
            )
            
            return BusSafetyMetrics(
                bus_id=bus_id,
                female_ratio=female_ratio,
                capacity_utilization=capacity_utilization,
                safety_score=safety_score,
                safety_level=safety_level,
                route_safety_score=route_safety_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating safety metrics: {e}")
            return BusSafetyMetrics(
                bus_id=bus_id,
                female_ratio=0.0,
                capacity_utilization=0.0,
                safety_score=0.0,
                safety_level=SafetyLevel.UNSAFE,
                route_safety_score=0.0,
                recommendations=["Error calculating safety metrics"]
            )
    
    def _generate_safety_recommendations(self, female_ratio: float, capacity_utilization: float, 
                                       route_safety_score: float, route_number: str) -> List[str]:
        """
        Generate safety recommendations based on current metrics
        
        Args:
            female_ratio: Ratio of female passengers
            capacity_utilization: Bus capacity utilization
            route_safety_score: Route safety score
            route_number: Current route number
            
        Returns:
            List of safety recommendations
        """
        recommendations = []
        
        # Female ratio recommendations
        if female_ratio < Config.MIN_FEMALE_RATIO_FOR_SAFETY:
            recommendations.append("Low female passenger ratio - consider waiting for more female passengers")
            recommendations.append("This route may not be safe for solo female travelers")
        else:
            recommendations.append("Good female passenger ratio - route is safe for female travelers")
        
        # Capacity recommendations
        if capacity_utilization > 0.8:
            recommendations.append("Bus is nearly full - consider waiting for next bus")
        elif capacity_utilization > 0.6:
            recommendations.append("Bus has moderate capacity - comfortable travel")
        else:
            recommendations.append("Bus has plenty of space - comfortable and safe travel")
        
        # Route-specific recommendations
        if route_safety_score < 0.7:
            recommendations.append(f"Route {route_number} has lower safety rating")
            
            # Suggest alternative routes
            alternative_routes = []
            for alt_route, route_info in Config.GURUGRAM_ROUTES.items():
                if route_info['safety_score'] > route_safety_score and alt_route != route_number:
                    alternative_routes.append(alt_route)
            
            if alternative_routes:
                recommendations.append(f"Consider alternative routes: {', '.join(alternative_routes)}")
        
        return recommendations
    
    def health_check(self) -> bool:
        """Check if the gender detection service is healthy"""
        try:
            # Check if cascades are loaded
            if self.face_cascade.empty() or self.eye_cascade.empty():
                return False
            
            # Test with a simple image
            test_image = np.zeros((100, 100), dtype=np.uint8)
            faces = self.face_cascade.detectMultiScale(test_image)
            
            return True
            
        except Exception as e:
            logger.error(f"Gender detection service health check failed: {e}")
            return False
