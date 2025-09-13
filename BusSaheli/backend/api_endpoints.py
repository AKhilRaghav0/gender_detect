#!/usr/bin/env python3
"""
REST API Endpoints for Bus Saheli
FastAPI-based endpoints for Flutter app integration
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime, timedelta
import uvicorn

from redis_integration import RedisService, BusData, SafetyLevel
from safety_algorithm import SafetyAlgorithm, SafetyScore
from unified_gender_detector import create_gender_detector, DetectionAlgorithm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bus Saheli API",
    description="Real-time bus safety monitoring API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
redis_service = RedisService()
safety_algorithm = SafetyAlgorithm()
gender_detector = create_gender_detector("insightface")  # Use InsightFace for best accuracy

# Pydantic models
class GenderDetectionRequest(BaseModel):
    bus_id: str
    image_data: str  # Base64 encoded image
    timestamp: Optional[datetime] = None

class GenderDetectionResponse(BaseModel):
    success: bool
    bus_id: str
    passenger_count: int
    female_count: int
    male_count: int
    safety_score: float
    safety_level: str
    recommendations: List[str]
    confidence: float
    processing_time: float

class BusDataResponse(BaseModel):
    bus_id: str
    route_number: str
    current_stop: str
    capacity: int
    passenger_count: int
    female_count: int
    male_count: int
    safety_score: float
    safety_level: str
    last_updated: datetime
    is_active: bool
    location: Optional[Dict[str, float]] = None

class RouteResponse(BaseModel):
    route_id: str
    route_name: str
    active_buses: List[BusDataResponse]
    average_safety_score: float
    total_passengers: int
    female_passengers: int

class SafetyAlertRequest(BaseModel):
    bus_id: str
    alert_type: str
    message: str
    severity: str = "medium"

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    redis_connected: bool
    active_routes: int
    total_buses: int
    system_uptime: str

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.bus_subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📱 New WebSocket connection: {len(self.active_connections)} total")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Remove from bus subscriptions
        for bus_id, connections in self.bus_subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
        
        logger.info(f"📱 WebSocket disconnected: {len(self.active_connections)} remaining")

    async def send_to_bus_subscribers(self, bus_id: str, data: dict):
        if bus_id in self.bus_subscriptions:
            for connection in self.bus_subscriptions[bus_id]:
                try:
                    await connection.send_text(json.dumps(data))
                except:
                    # Remove broken connections
                    self.bus_subscriptions[bus_id].remove(connection)

    def subscribe_to_bus(self, websocket: WebSocket, bus_id: str):
        if bus_id not in self.bus_subscriptions:
            self.bus_subscriptions[bus_id] = []
        self.bus_subscriptions[bus_id].append(websocket)
        logger.info(f"📱 Subscribed to bus {bus_id}")

manager = ConnectionManager()

# API Endpoints

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """System health check endpoint"""
    try:
        # Check Redis connection
        redis_connected = redis_service.redis_client.ping()
        
        # Get system stats
        stats = redis_service.get_system_stats()
        
        return SystemHealthResponse(
            status="healthy" if redis_connected else "degraded",
            timestamp=datetime.now(),
            redis_connected=redis_connected,
            active_routes=stats.get('active_routes', 0),
            total_buses=stats.get('total_buses', 0),
            system_uptime="24h"  # Would calculate actual uptime
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/v1/detect-gender", response_model=GenderDetectionResponse)
async def detect_gender(request: GenderDetectionRequest):
    """Process camera frame for gender detection"""
    start_time = datetime.now()
    
    try:
        # Implement actual gender detection using unified detector
        import base64
        import numpy as np
        import cv2
        
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use unified detector for actual gender detection
        detection_results = gender_detector.detect_gender(image)
        
        # Count passengers by gender
        passenger_count = len(detection_results)
        female_count = sum(1 for result in detection_results if result.gender == "Female")
        male_count = sum(1 for result in detection_results if result.gender == "Male")
        
        # Calculate average confidence
        avg_confidence = sum(result.confidence for result in detection_results) / len(detection_results) if detection_results else 0.0
        
        # Create bus data
        bus_data = BusData(
            bus_id=request.bus_id,
            route_number="118",  # Would get from request
            current_stop="Gurgaon Metro",
            capacity=50,
            passenger_count=passenger_count,
            female_count=female_count,
            male_count=male_count,
            safety_score=0.0,  # Will be calculated
            safety_level=SafetyLevel.SAFE,
            last_updated=datetime.now(),
            is_active=True
        )
        
        # Calculate safety score
        safety_score = safety_algorithm.calculate_safety_score(bus_data)
        
        # Update bus data with safety score
        bus_data.safety_score = safety_score.overall_score
        bus_data.safety_level = safety_score.safety_level
        
        # Publish to Redis
        redis_service.publish_bus_data(bus_data)
        
        # Send to WebSocket subscribers
        await manager.send_to_bus_subscribers(request.bus_id, {
            "type": "bus_update",
            "data": {
                "bus_id": bus_data.bus_id,
                "passenger_count": bus_data.passenger_count,
                "female_count": bus_data.female_count,
                "male_count": bus_data.male_count,
                "safety_score": bus_data.safety_score,
                "safety_level": bus_data.safety_level.value,
                "timestamp": bus_data.last_updated.isoformat()
            }
        })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return GenderDetectionResponse(
            success=True,
            bus_id=request.bus_id,
            passenger_count=passenger_count,
            female_count=female_count,
            male_count=male_count,
            safety_score=safety_score.overall_score,
            safety_level=safety_score.safety_level.value,
            recommendations=safety_score.recommendations,
            confidence=avg_confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Gender detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/api/v1/routes", response_model=List[RouteResponse])
async def get_active_routes():
    """Get all active bus routes"""
    try:
        active_routes = redis_service.get_active_routes()
        routes = []
        
        for route_id in active_routes:
            buses = redis_service.get_route_buses(route_id)
            
            if buses:
                total_passengers = sum(bus.passenger_count for bus in buses)
                female_passengers = sum(bus.female_count for bus in buses)
                avg_safety_score = sum(bus.safety_score for bus in buses) / len(buses)
                
                # Convert buses to response format
                bus_responses = []
                for bus in buses:
                    bus_responses.append(BusDataResponse(
                        bus_id=bus.bus_id,
                        route_number=bus.route_number,
                        current_stop=bus.current_stop,
                        capacity=bus.capacity,
                        passenger_count=bus.passenger_count,
                        female_count=bus.female_count,
                        male_count=bus.male_count,
                        safety_score=bus.safety_score,
                        safety_level=bus.safety_level.value,
                        last_updated=bus.last_updated,
                        is_active=bus.is_active,
                        location=bus.location
                    ))
                
                routes.append(RouteResponse(
                    route_id=route_id,
                    route_name=f"Route {route_id}",
                    active_buses=bus_responses,
                    average_safety_score=avg_safety_score,
                    total_passengers=total_passengers,
                    female_passengers=female_passengers
                ))
        
        return routes
        
    except Exception as e:
        logger.error(f"❌ Failed to get active routes: {e}")
        raise HTTPException(status_code=500, detail="Failed to get routes")

@app.get("/api/v1/buses/{bus_id}", response_model=BusDataResponse)
async def get_bus_data(bus_id: str):
    """Get current data for a specific bus"""
    try:
        bus_data = redis_service.get_bus_data(bus_id)
        
        if not bus_data:
            raise HTTPException(status_code=404, detail="Bus not found")
        
        return BusDataResponse(
            bus_id=bus_data.bus_id,
            route_number=bus_data.route_number,
            current_stop=bus_data.current_stop,
            capacity=bus_data.capacity,
            passenger_count=bus_data.passenger_count,
            female_count=bus_data.female_count,
            male_count=bus_data.male_count,
            safety_score=bus_data.safety_score,
            safety_level=bus_data.safety_level.value,
            last_updated=bus_data.last_updated,
            is_active=bus_data.is_active,
            location=bus_data.location
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get bus data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get bus data")

@app.get("/api/v1/routes/{route_id}/buses", response_model=List[BusDataResponse])
async def get_route_buses(route_id: str):
    """Get all buses for a specific route"""
    try:
        buses = redis_service.get_route_buses(route_id)
        
        bus_responses = []
        for bus in buses:
            bus_responses.append(BusDataResponse(
                bus_id=bus.bus_id,
                route_number=bus.route_number,
                current_stop=bus.current_stop,
                capacity=bus.capacity,
                passenger_count=bus.passenger_count,
                female_count=bus.female_count,
                male_count=bus.male_count,
                safety_score=bus.safety_score,
                safety_level=bus.safety_level.value,
                last_updated=bus.last_updated,
                is_active=bus.is_active,
                location=bus.location
            ))
        
        return bus_responses
        
    except Exception as e:
        logger.error(f"❌ Failed to get route buses: {e}")
        raise HTTPException(status_code=500, detail="Failed to get route buses")

@app.post("/api/v1/alerts")
async def create_safety_alert(request: SafetyAlertRequest):
    """Create a safety alert"""
    try:
        alert_data = {
            "bus_id": request.bus_id,
            "alert_type": request.alert_type,
            "message": request.message,
            "severity": request.severity,
            "timestamp": datetime.now().isoformat()
        }
        
        success = redis_service.add_safety_alert(alert_data)
        
        if success:
            # Send alert to all connected clients
            for connection in manager.active_connections:
                try:
                    await connection.send_text(json.dumps({
                        "type": "safety_alert",
                        "data": alert_data
                    }))
                except:
                    pass
            
            return {"success": True, "message": "Alert created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create alert")
            
    except Exception as e:
        logger.error(f"❌ Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to create alert")

@app.get("/api/v1/alerts")
async def get_safety_alerts(limit: int = 10):
    """Get recent safety alerts"""
    try:
        alerts = redis_service.get_safety_alerts(limit)
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"❌ Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

@app.get("/api/v1/detector/status")
async def get_detector_status():
    """Get current detector status and performance"""
    try:
        stats = gender_detector.get_performance_stats()
        return {
            "algorithm": gender_detector.algorithm.value,
            "performance_stats": stats,
            "status": "active"
        }
    except Exception as e:
        logger.error(f"❌ Failed to get detector status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get detector status")

@app.post("/api/v1/detector/switch")
async def switch_detector_algorithm(algorithm: str):
    """Switch to a different detection algorithm"""
    try:
        # Validate algorithm
        valid_algorithms = [algo.value for algo in DetectionAlgorithm]
        if algorithm not in valid_algorithms:
            raise HTTPException(status_code=400, detail=f"Invalid algorithm. Valid options: {valid_algorithms}")
        
        # Switch algorithm
        gender_detector.switch_algorithm(DetectionAlgorithm(algorithm))
        
        return {
            "success": True,
            "message": f"Switched to {algorithm} algorithm",
            "current_algorithm": gender_detector.algorithm.value
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to switch algorithm: {e}")
        raise HTTPException(status_code=500, detail="Failed to switch algorithm")

@app.get("/api/v1/statistics")
async def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        # Get Redis stats
        redis_stats = redis_service.get_system_stats()
        
        # Get detector stats
        detector_stats = gender_detector.get_performance_stats()
        
        # Get active routes
        active_routes = redis_service.get_active_routes()
        
        return {
            "redis_stats": redis_stats,
            "detector_stats": detector_stats,
            "active_routes": len(active_routes),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.websocket("/ws/bus-safety")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe_bus":
                bus_id = message.get("bus_id")
                if bus_id:
                    manager.subscribe_to_bus(websocket, bus_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "bus_id": bus_id
                    }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Bus Saheli API starting up...")
    
    # Test Redis connection
    try:
        redis_service.redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    
    logger.info("🚀 API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Bus Saheli API shutting down...")
    
    # Close all WebSocket connections
    for connection in manager.active_connections:
        try:
            await connection.close()
        except:
            pass
    
    logger.info("🛑 API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "api_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
