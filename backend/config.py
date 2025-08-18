"""
Configuration file for Bus Safety Gender Detection System Backend
Contains Appwrite credentials and system settings
"""

import os
from typing import Optional

class Config:
    """Configuration class for the backend system"""
    
    # Appwrite Configuration
    APPWRITE_PROJECT_ID = "68a1c6da0032d8f8c2f7"
    APPWRITE_PUBLIC_ENDPOINT = "https://appwrite.akhilraghav.codes/v1"
    APPWRITE_API_KEY = "standard_629a9c24f548d3da09f59a1353055e44656045d87dbf07eb1e9b06795332abefb61a936b5234d7e0751cf4f6e34fbf6ad961fc3b93b45f99e19d3aacd5a4e5f98ff1fdd10b5e9b052a49bddcf278aa2d66a4f910ab8235598b86d58870dfd5dc5a12e968bece2e8d7775f86f229f17b6bb399f61c938c44c7f9c5614ec7086f5"
    
    # Database Collections
    DATABASE_ID = "bus_safety_system"  # Main database ID
    BUS_COLLECTION_ID = "buses"
    GENDER_COUNT_COLLECTION_ID = "gender_counts"
    ROUTE_COLLECTION_ID = "routes"
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_DEBUG = True
    
    # Gender Detection Settings
    DETECTION_CONFIDENCE_THRESHOLD = 0.6
    MAX_FACES_PER_FRAME = 10
    FRAME_PROCESSING_INTERVAL = 1.0  # seconds
    
    # Bus Safety Settings
    MIN_FEMALE_RATIO_FOR_SAFETY = 0.3  # 30% females for safe travel
    MAX_BUS_CAPACITY = 50  # Maximum passengers per bus
    SAFETY_SCORE_WEIGHTS = {
        "female_ratio": 0.4,
        "capacity_utilization": 0.3,
        "route_safety": 0.3
    }
    
    # Route Information (Gurugram)
    GURUGRAM_ROUTES = {
        "118": {"name": "Route 118", "safety_score": 0.8, "stops": ["Cyber City", "Huda City Center", "MG Road"]},
        "111": {"name": "Route 111", "safety_score": 0.7, "stops": ["Sector 29", "Old Delhi Road", "Railway Station"]},
        "218": {"name": "Route 218", "safety_score": 0.9, "stops": ["Sector 56", "NH8", "Airport Road"]}
    }
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = "backend.log"
    
    @classmethod
    def get_appwrite_config(cls) -> dict:
        """Get Appwrite configuration as dictionary"""
        return {
            "project_id": cls.APPWRITE_PROJECT_ID,
            "endpoint": cls.APPWRITE_PUBLIC_ENDPOINT,
            "api_key": cls.APPWRITE_API_KEY
        }
    
    @classmethod
    def get_api_config(cls) -> dict:
        """Get API configuration as dictionary"""
        return {
            "host": cls.API_HOST,
            "port": cls.API_PORT,
            "debug": cls.API_DEBUG
        }
