"""
Data models for Bus Safety Gender Detection System
Defines the structure of bus data, gender counts, and route information
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class Gender(str, Enum):
    """Gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class BusStatus(str, Enum):
    """Bus status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class SafetyLevel(str, Enum):
    """Safety level enumeration"""
    SAFE = "safe"
    MODERATE = "moderate"
    UNSAFE = "unsafe"

class BusInfo(BaseModel):
    """Bus information model"""
    bus_id: str = Field(..., description="Unique bus identifier")
    route_number: str = Field(..., description="Bus route number (e.g., 118, 111, 218)")
    driver_name: Optional[str] = Field(None, description="Driver's name")
    driver_contact: Optional[str] = Field(None, description="Driver's contact number")
    capacity: int = Field(50, description="Maximum passenger capacity")
    current_location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates {lat, lng}")
    status: BusStatus = Field(BusStatus.ACTIVE, description="Current bus status")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    is_active: bool = Field(True, description="Whether bus is currently in service")
    
    class Config:
        schema_extra = {
            "example": {
                "bus_id": "BUS_001",
                "route_number": "118",
                "driver_name": "Rajesh Kumar",
                "driver_contact": "+91-98765-43210",
                "capacity": 50,
                "current_location": {"lat": 28.4595, "lng": 77.0266},
                "status": "active",
                "is_active": True
            }
        }

class GenderCount(BaseModel):
    """Real-time gender count model"""
    bus_id: str = Field(..., description="Bus identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Count timestamp")
    male_count: int = Field(0, ge=0, description="Number of male passengers")
    female_count: int = Field(0, ge=0, description="Number of female passengers")
    total_passengers: int = Field(0, ge=0, description="Total passenger count")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence")
    
    class Config:
        schema_extra = {
            "example": {
                "bus_id": "BUS_001",
                "male_count": 15,
                "female_count": 8,
                "total_passengers": 23,
                "confidence_score": 0.85
            }
        }

class BusSafetyMetrics(BaseModel):
    """Bus safety metrics and calculations"""
    bus_id: str = Field(..., description="Bus identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    female_ratio: float = Field(0.0, ge=0.0, le=1.0, description="Ratio of female passengers")
    capacity_utilization: float = Field(0.0, ge=0.0, le=1.0, description="Bus capacity utilization")
    safety_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall safety score")
    safety_level: SafetyLevel = Field(SafetyLevel.MODERATE, description="Safety level classification")
    route_safety_score: float = Field(0.0, ge=0.0, le=1.0, description="Route-specific safety score")
    recommendations: List[str] = Field(default_factory=list, description="Safety recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "bus_id": "BUS_001",
                "female_ratio": 0.35,
                "capacity_utilization": 0.46,
                "safety_score": 0.72,
                "safety_level": "safe",
                "route_safety_score": 0.8,
                "recommendations": ["Route is safe for female passengers", "Consider alternative route 218 for better safety"]
            }
        }

class RouteInfo(BaseModel):
    """Route information model"""
    route_number: str = Field(..., description="Route number")
    route_name: str = Field(..., description="Route name")
    safety_score: float = Field(0.0, ge=0.0, le=1.0, description="Route safety score")
    stops: List[str] = Field(default_factory=list, description="Bus stops on this route")
    alternative_routes: List[str] = Field(default_factory=list, description="Alternative route numbers")
    is_active: bool = Field(True, description="Whether route is currently active")
    
    class Config:
        schema_extra = {
            "example": {
                "route_number": "118",
                "route_name": "Route 118",
                "safety_score": 0.8,
                "stops": ["Cyber City", "Huda City Center", "MG Road"],
                "alternative_routes": ["111", "218"]
            }
        }

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    error_code: Optional[str] = Field(None, description="Error code if applicable")

class GenderDetectionRequest(BaseModel):
    """Request model for gender detection"""
    bus_id: str = Field(..., description="Bus identifier")
    image_data: str = Field(..., description="Base64 encoded image data")
    timestamp: Optional[datetime] = Field(None, description="Detection timestamp")
    
class GenderDetectionResponse(BaseModel):
    """Response model for gender detection"""
    bus_id: str = Field(..., description="Bus identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")
    detected_faces: int = Field(0, description="Number of faces detected")
    gender_counts: Dict[str, int] = Field(default_factory=dict, description="Gender count breakdown")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores per gender")
    processing_time: float = Field(0.0, description="Processing time in seconds")
