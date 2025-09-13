#!/usr/bin/env python3
"""
Safety Algorithm for Bus Saheli
Calculates safety scores and provides recommendations for bus travel
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    UNSAFE = "UNSAFE"

class CrowdDensity(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    OVERCROWDED = "OVERCROWDED"

@dataclass
class SafetyScore:
    overall_score: float
    safety_level: SafetyLevel
    female_ratio_score: float
    capacity_score: float
    crowd_density_score: float
    historical_score: float
    recommendations: List[str]
    confidence: float

@dataclass
class BusData:
    bus_id: str
    route_number: str
    capacity: int
    passenger_count: int
    female_count: int
    male_count: int
    timestamp: datetime

class SafetyAlgorithm:
    """Advanced safety algorithm for bus travel assessment"""
    
    def __init__(self):
        # Safety thresholds
        self.female_ratio_thresholds = {
            'safe': 0.4,      # 40%+ female passengers
            'moderate': 0.2,  # 20-40% female passengers
            'unsafe': 0.0     # <20% female passengers
        }
        
        self.capacity_thresholds = {
            'low': 0.3,       # <30% capacity
            'normal': 0.7,    # 30-70% capacity
            'high': 0.9,      # 70-90% capacity
            'overcrowded': 1.0 # >90% capacity
        }
        
        # Scoring weights
        self.weights = {
            'female_ratio': 0.40,    # 40% weight
            'capacity_ratio': 0.30,  # 30% weight
            'crowd_density': 0.20,   # 20% weight
            'historical': 0.10       # 10% weight
        }
        
        # Historical data (in real implementation, this would come from database)
        self.historical_data = {}
        
        logger.info("🛡️ Safety Algorithm initialized")
    
    def calculate_safety_score(self, bus_data: BusData, historical_data: Optional[Dict] = None) -> SafetyScore:
        """Calculate comprehensive safety score for a bus"""
        try:
            # Calculate individual scores
            female_ratio_score = self._calculate_female_ratio_score(bus_data)
            capacity_score = self._calculate_capacity_score(bus_data)
            crowd_density_score = self._calculate_crowd_density_score(bus_data)
            historical_score = self._calculate_historical_score(bus_data, historical_data)
            
            # Calculate weighted overall score
            overall_score = (
                female_ratio_score * self.weights['female_ratio'] +
                capacity_score * self.weights['capacity_ratio'] +
                crowd_density_score * self.weights['crowd_density'] +
                historical_score * self.weights['historical']
            )
            
            # Determine safety level
            safety_level = self._determine_safety_level(overall_score, bus_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                overall_score, safety_level, bus_data
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(bus_data)
            
            return SafetyScore(
                overall_score=overall_score,
                safety_level=safety_level,
                female_ratio_score=female_ratio_score,
                capacity_score=capacity_score,
                crowd_density_score=crowd_density_score,
                historical_score=historical_score,
                recommendations=recommendations,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating safety score: {e}")
            return self._get_default_safety_score()
    
    def _calculate_female_ratio_score(self, bus_data: BusData) -> float:
        """Calculate score based on female passenger ratio"""
        if bus_data.passenger_count == 0:
            return 0.5  # Neutral score for empty bus
        
        female_ratio = bus_data.female_count / bus_data.passenger_count
        
        # Higher female ratio = higher safety score
        if female_ratio >= self.female_ratio_thresholds['safe']:
            return 1.0
        elif female_ratio >= self.female_ratio_thresholds['moderate']:
            # Linear interpolation between 0.5 and 1.0
            ratio = (female_ratio - self.female_ratio_thresholds['moderate']) / \
                   (self.female_ratio_thresholds['safe'] - self.female_ratio_thresholds['moderate'])
            return 0.5 + (ratio * 0.5)
        else:
            # Linear interpolation between 0.0 and 0.5
            ratio = female_ratio / self.female_ratio_thresholds['moderate']
            return ratio * 0.5
    
    def _calculate_capacity_score(self, bus_data: BusData) -> float:
        """Calculate score based on bus capacity utilization"""
        if bus_data.capacity == 0:
            return 0.5
        
        capacity_ratio = bus_data.passenger_count / bus_data.capacity
        
        # Optimal capacity is 50-70%
        if 0.5 <= capacity_ratio <= 0.7:
            return 1.0
        elif 0.3 <= capacity_ratio < 0.5:
            # Lower capacity = slightly lower score
            return 0.8 + (capacity_ratio - 0.3) * 1.0
        elif 0.7 < capacity_ratio <= 0.9:
            # Higher capacity = lower score
            return 1.0 - (capacity_ratio - 0.7) * 2.5
        else:
            # Very low or very high capacity = low score
            return 0.3
    
    def _calculate_crowd_density_score(self, bus_data: BusData) -> float:
        """Calculate score based on crowd density"""
        if bus_data.capacity == 0:
            return 0.5
        
        capacity_ratio = bus_data.passenger_count / bus_data.capacity
        
        # Determine crowd density level
        if capacity_ratio < self.capacity_thresholds['low']:
            density = CrowdDensity.LOW
        elif capacity_ratio < self.capacity_thresholds['normal']:
            density = CrowdDensity.NORMAL
        elif capacity_ratio < self.capacity_thresholds['high']:
            density = CrowdDensity.HIGH
        else:
            density = CrowdDensity.OVERCROWDED
        
        # Score based on density
        density_scores = {
            CrowdDensity.LOW: 0.6,        # Too empty can feel unsafe
            CrowdDensity.NORMAL: 1.0,     # Optimal density
            CrowdDensity.HIGH: 0.7,       # Crowded but manageable
            CrowdDensity.OVERCROWDED: 0.2  # Very unsafe
        }
        
        return density_scores[density]
    
    def _calculate_historical_score(self, bus_data: BusData, historical_data: Optional[Dict]) -> float:
        """Calculate score based on historical safety data"""
        if not historical_data:
            return 0.5  # Neutral score without historical data
        
        route_key = bus_data.route_number
        
        # Get historical safety data for this route
        if route_key not in self.historical_data:
            return 0.5
        
        historical = self.historical_data[route_key]
        
        # Calculate average safety score for this route
        if 'safety_scores' in historical and len(historical['safety_scores']) > 0:
            avg_score = sum(historical['safety_scores']) / len(historical['safety_scores'])
            return avg_score
        
        return 0.5
    
    def _determine_safety_level(self, overall_score: float, bus_data: BusData) -> SafetyLevel:
        """Determine safety level based on overall score and additional factors"""
        # Base safety level on score
        if overall_score >= 0.8:
            base_level = SafetyLevel.SAFE
        elif overall_score >= 0.5:
            base_level = SafetyLevel.MODERATE
        else:
            base_level = SafetyLevel.UNSAFE
        
        # Adjust based on specific conditions
        if bus_data.passenger_count == 0:
            return SafetyLevel.MODERATE  # Empty bus is moderately safe
        
        female_ratio = bus_data.female_count / bus_data.passenger_count
        
        # Override to UNSAFE if very low female ratio
        if female_ratio < 0.1 and bus_data.passenger_count > 10:
            return SafetyLevel.UNSAFE
        
        # Override to UNSAFE if overcrowded
        if bus_data.passenger_count > bus_data.capacity * 0.95:
            return SafetyLevel.UNSAFE
        
        return base_level
    
    def _generate_recommendations(self, overall_score: float, safety_level: SafetyLevel, bus_data: BusData) -> List[str]:
        """Generate safety recommendations based on current conditions"""
        recommendations = []
        
        # Female ratio recommendations
        if bus_data.passenger_count > 0:
            female_ratio = bus_data.female_count / bus_data.passenger_count
            
            if female_ratio < 0.2:
                recommendations.append("⚠️ Very few female passengers - consider waiting for next bus")
            elif female_ratio < 0.4:
                recommendations.append("⚠️ Low female passenger count - be extra cautious")
            else:
                recommendations.append("✅ Good female passenger representation")
        
        # Capacity recommendations
        capacity_ratio = bus_data.passenger_count / bus_data.capacity if bus_data.capacity > 0 else 0
        
        if capacity_ratio > 0.9:
            recommendations.append("🚫 Bus is overcrowded - avoid if possible")
        elif capacity_ratio > 0.7:
            recommendations.append("⚠️ Bus is quite full - consider next bus")
        elif capacity_ratio < 0.3:
            recommendations.append("⚠️ Bus is quite empty - sit near other passengers")
        else:
            recommendations.append("✅ Good passenger density")
        
        # Time-based recommendations
        current_hour = datetime.now().hour
        if 22 <= current_hour or current_hour <= 5:
            recommendations.append("🌙 Night travel - stay alert and sit near other passengers")
        
        # General safety recommendations
        if safety_level == SafetyLevel.UNSAFE:
            recommendations.append("🚨 Consider alternative transportation")
            recommendations.append("📱 Share your location with trusted contacts")
        elif safety_level == SafetyLevel.MODERATE:
            recommendations.append("👀 Stay alert and aware of surroundings")
            recommendations.append("📱 Keep phone charged and accessible")
        else:
            recommendations.append("✅ Safe conditions for travel")
        
        return recommendations
    
    def _calculate_confidence(self, bus_data: BusData) -> float:
        """Calculate confidence in the safety assessment"""
        confidence = 1.0
        
        # Reduce confidence for very low passenger counts
        if bus_data.passenger_count < 5:
            confidence *= 0.7
        
        # Reduce confidence for very high passenger counts (harder to count accurately)
        if bus_data.passenger_count > bus_data.capacity * 0.9:
            confidence *= 0.8
        
        # Reduce confidence if no female passengers (might be detection error)
        if bus_data.passenger_count > 10 and bus_data.female_count == 0:
            confidence *= 0.6
        
        return min(1.0, max(0.1, confidence))
    
    def _get_default_safety_score(self) -> SafetyScore:
        """Return default safety score in case of errors"""
        return SafetyScore(
            overall_score=0.5,
            safety_level=SafetyLevel.MODERATE,
            female_ratio_score=0.5,
            capacity_score=0.5,
            crowd_density_score=0.5,
            historical_score=0.5,
            recommendations=["⚠️ Unable to assess safety - use caution"],
            confidence=0.1
        )
    
    def update_historical_data(self, route_number: str, safety_score: float):
        """Update historical safety data for a route"""
        if route_number not in self.historical_data:
            self.historical_data[route_number] = {
                'safety_scores': [],
                'last_updated': datetime.now()
            }
        
        # Add new score
        self.historical_data[route_number]['safety_scores'].append(safety_score)
        
        # Keep only last 100 scores
        if len(self.historical_data[route_number]['safety_scores']) > 100:
            self.historical_data[route_number]['safety_scores'] = \
                self.historical_data[route_number]['safety_scores'][-100:]
        
        self.historical_data[route_number]['last_updated'] = datetime.now()
        
        logger.info(f"📊 Updated historical data for route {route_number}")

# Example usage
if __name__ == "__main__":
    # Initialize safety algorithm
    safety_algo = SafetyAlgorithm()
    
    # Create sample bus data
    sample_bus = BusData(
        bus_id="BUS_118_001",
        route_number="118",
        capacity=50,
        passenger_count=35,
        female_count=15,
        male_count=20,
        timestamp=datetime.now()
    )
    
    # Calculate safety score
    safety_score = safety_algo.calculate_safety_score(sample_bus)
    
    print(f"Safety Score: {safety_score.overall_score:.2f}")
    print(f"Safety Level: {safety_score.safety_level.value}")
    print(f"Female Ratio Score: {safety_score.female_ratio_score:.2f}")
    print(f"Capacity Score: {safety_score.capacity_score:.2f}")
    print(f"Confidence: {safety_score.confidence:.2f}")
    print("Recommendations:")
    for rec in safety_score.recommendations:
        print(f"  - {rec}")
