"""
Main FastAPI application for Bus Safety Gender Detection System
Provides REST API endpoints for gender detection, bus management, and safety metrics
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Config
from models import (
    BusInfo, GenderCount, BusSafetyMetrics, RouteInfo, 
    APIResponse, GenderDetectionRequest, GenderDetectionResponse
)
from appwrite_service import AppwriteService
from gender_detection_service import GenderDetectionService

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bus Safety Gender Detection System API",
    description="Real-time gender detection and bus safety monitoring system for Gurugram bus routes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instances
appwrite_service: Optional[AppwriteService] = None
gender_detection_service: Optional[GenderDetectionService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global appwrite_service, gender_detection_service
    
    try:
        logger.info("Starting Bus Safety Gender Detection System...")
        
        # Initialize Appwrite service
        appwrite_service = AppwriteService()
        logger.info("Appwrite service initialized")
        
        # Initialize gender detection service
        gender_detection_service = GenderDetectionService()
        logger.info("Gender detection service initialized")
        
        # Setup database collections
        appwrite_service.create_database_collections()
        logger.info("Database setup completed")
        
        logger.info("System startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global appwrite_service
    
    try:
        if appwrite_service:
            appwrite_service.close()
        logger.info("System shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Dependency functions
def get_appwrite_service() -> AppwriteService:
    """Get Appwrite service instance"""
    if not appwrite_service:
        raise HTTPException(status_code=503, detail="Appwrite service not available")
    return appwrite_service

def get_gender_detection_service() -> GenderDetectionService:
    """Get gender detection service instance"""
    if not gender_detection_service:
        raise HTTPException(status_code=503, detail="Gender detection service not available")
    return gender_detection_service

# Health Check Endpoints
@app.get("/health", response_model=APIResponse)
async def health_check():
    """System health check endpoint"""
    try:
        appwrite_healthy = appwrite_service.health_check() if appwrite_service else False
        detection_healthy = gender_detection_service.health_check() if gender_detection_service else False
        
        overall_health = appwrite_healthy and detection_healthy
        
        return APIResponse(
            success=overall_health,
            message="System health check completed",
            data={
                "overall_health": overall_health,
                "appwrite_service": appwrite_healthy,
                "gender_detection_service": detection_healthy,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return APIResponse(
            success=False,
            message="Health check failed",
            error_code="HEALTH_CHECK_ERROR"
        )

@app.get("/test", response_model=APIResponse)
async def test_endpoint():
    """Simple test endpoint that doesn't require external services"""
    try:
        return APIResponse(
            success=True,
            message="Test endpoint working correctly",
            data={
                "timestamp": datetime.utcnow().isoformat(),
                "message": "FastAPI is working! 🚀"
            }
        )
        
    except Exception as e:
        logger.error(f"Test endpoint failed: {e}")
        return APIResponse(
            success=False,
            message="Test endpoint failed",
            error_code="TEST_ERROR"
        )

@app.get("/test-detection", response_model=APIResponse)
async def test_detection_endpoint():
    """Test gender detection service without database operations"""
    try:
        if not gender_detection_service:
            return APIResponse(
                success=False,
                message="Gender detection service not available",
                error_code="SERVICE_UNAVAILABLE"
            )
        
        # Test if the service is healthy
        is_healthy = gender_detection_service.health_check()
        
        return APIResponse(
            success=is_healthy,
            message="Gender detection service test completed",
            data={
                "timestamp": datetime.utcnow().isoformat(),
                "service_healthy": is_healthy,
                "message": "Detection service test completed"
            }
        )
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        return APIResponse(
            success=False,
            message="Detection test failed",
            error_code="DETECTION_TEST_ERROR"
        )

# Gender Detection Endpoints
@app.post("/detect-gender", response_model=GenderDetectionResponse)
async def detect_gender(
    request: GenderDetectionRequest,
    background_tasks: BackgroundTasks,
    appwrite: AppwriteService = Depends(get_appwrite_service),
    detection: GenderDetectionService = Depends(get_gender_detection_service)
):
    """Process image for gender detection and save results"""
    try:
        logger.info(f"Processing gender detection for bus {request.bus_id}")
        
        # Process image for gender detection
        total_faces, gender_counts, confidence_scores = detection.process_image(request.image_data)
        
        # Create gender count object
        overall_confidence = max(confidence_scores.values()) if confidence_scores else 0.0
        gender_count = detection.create_gender_count(request.bus_id, gender_counts, overall_confidence)
        
        # Calculate safety metrics
        bus_info = appwrite.get_bus_info(request.bus_id)
        route_number = bus_info.route_number if bus_info else "unknown"
        safety_metrics = detection.calculate_safety_metrics(request.bus_id, gender_count, route_number)
        
        # Save to database in background
        background_tasks.add_task(appwrite.save_gender_count, gender_count)
        
        # Prepare response
        response = GenderDetectionResponse(
            bus_id=request.bus_id,
            detected_faces=total_faces,
            gender_counts=gender_counts,
            confidence_scores=confidence_scores,
            processing_time=0.0  # Could be calculated if needed
        )
        
        logger.info(f"Gender detection completed for bus {request.bus_id}: {gender_counts}")
        return response
        
    except Exception as e:
        logger.error(f"Gender detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gender detection failed: {str(e)}")

# Bus Management Endpoints
@app.post("/buses/register", response_model=APIResponse)
async def register_bus(
    bus_info: BusInfo,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Register a new bus in the system"""
    try:
        success = appwrite.register_bus(bus_info)
        
        if success:
            return APIResponse(
                success=True,
                message=f"Bus {bus_info.bus_id} registered successfully",
                data=bus_info.model_dump() if hasattr(bus_info, 'model_dump') else bus_info.dict()
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to register bus")
            
    except Exception as e:
        logger.error(f"Bus registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bus registration failed: {str(e)}")

@app.get("/buses/{bus_id}", response_model=BusInfo)
async def get_bus_info(
    bus_id: str,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Get bus information by ID"""
    try:
        bus_info = appwrite.get_bus_info(bus_id)
        
        if not bus_info:
            raise HTTPException(status_code=404, detail=f"Bus {bus_id} not found")
            
        return bus_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get bus info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get bus info: {str(e)}")

@app.put("/buses/{bus_id}/status", response_model=APIResponse)
async def update_bus_status(
    bus_id: str,
    status: str,
    location: Optional[dict] = None,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Update bus status and location"""
    try:
        success = appwrite.update_bus_status(bus_id, status, location)
        
        if success:
            return APIResponse(
                success=True,
                message=f"Bus {bus_id} status updated to {status}",
                data={"bus_id": bus_id, "status": status, "location": location}
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to update bus status")
            
    except Exception as e:
        logger.error(f"Failed to update bus status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update bus status: {str(e)}")

# Gender Count Endpoints
@app.get("/buses/{bus_id}/gender-count", response_model=GenderCount)
async def get_latest_gender_count(
    bus_id: str,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Get the latest gender count for a bus"""
    try:
        gender_count = appwrite.get_latest_gender_count(bus_id)
        
        if not gender_count:
            raise HTTPException(status_code=404, detail=f"No gender count data found for bus {bus_id}")
            
        return gender_count
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get gender count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get gender count: {str(e)}")

@app.get("/buses/{bus_id}/gender-count/history", response_model=List[GenderCount])
async def get_gender_count_history(
    bus_id: str,
    hours: int = 24,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Get gender count history for a bus"""
    try:
        history = appwrite.get_gender_count_history(bus_id, hours)
        return history
        
    except Exception as e:
        logger.error(f"Failed to get gender count history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get gender count history: {str(e)}")

# Safety Metrics Endpoints
@app.get("/buses/{bus_id}/safety-metrics", response_model=BusSafetyMetrics)
async def get_bus_safety_metrics(
    bus_id: str,
    appwrite: AppwriteService = Depends(get_appwrite_service),
    detection: GenderDetectionService = Depends(get_gender_detection_service)
):
    """Get current safety metrics for a bus"""
    try:
        # Get latest gender count
        gender_count = appwrite.get_latest_gender_count(bus_id)
        if not gender_count:
            raise HTTPException(status_code=404, detail=f"No gender count data found for bus {bus_id}")
        
        # Get bus info for route number
        bus_info = appwrite.get_bus_info(bus_id)
        route_number = bus_info.route_number if bus_info else "unknown"
        
        # Calculate safety metrics
        safety_metrics = detection.calculate_safety_metrics(bus_id, gender_count, route_number)
        return safety_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get safety metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get safety metrics: {str(e)}")

# Route Information Endpoints
@app.get("/routes/{route_number}", response_model=RouteInfo)
async def get_route_info(
    route_number: str,
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Get route information by route number"""
    try:
        route_info = appwrite.get_route_info(route_number)
        
        if not route_info:
            # Return default route info from config
            default_route = Config.GURUGRAM_ROUTES.get(route_number)
            if default_route:
                return RouteInfo(
                    route_number=route_number,
                    route_name=default_route["name"],
                    safety_score=default_route["safety_score"],
                    stops=default_route["stops"],
                    alternative_routes=[r for r in Config.GURUGRAM_ROUTES.keys() if r != route_number]
                )
            else:
                raise HTTPException(status_code=404, detail=f"Route {route_number} not found")
            
        return route_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get route info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get route info: {str(e)}")

@app.get("/routes", response_model=List[RouteInfo])
async def get_all_routes(
    appwrite: AppwriteService = Depends(get_appwrite_service)
):
    """Get all available routes"""
    try:
        routes = appwrite.get_all_routes()
        
        if not routes:
            # Return default routes from config
            routes = []
            for route_num, route_info in Config.GURUGRAM_ROUTES.items():
                routes.append(RouteInfo(
                    route_number=route_num,
                    route_name=route_info["name"],
                    safety_score=route_info["safety_score"],
                    stops=route_info["stops"],
                    alternative_routes=[r for r in Config.GURUGRAM_ROUTES.keys() if r != route_num]
                ))
        
        return routes
        
    except Exception as e:
        logger.error(f"Failed to get all routes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get all routes: {str(e)}")

# System Information Endpoints
@app.get("/system/info", response_model=APIResponse)
async def get_system_info():
    """Get system information and configuration"""
    try:
        system_info = {
            "system_name": "Bus Safety Gender Detection System",
            "version": "1.0.0",
            "location": "Gurugram, India",
            "supported_routes": list(Config.GURUGRAM_ROUTES.keys()),
            "max_bus_capacity": Config.MAX_BUS_CAPACITY,
            "min_female_ratio_for_safety": Config.MIN_FEMALE_RATIO_FOR_SAFETY,
            "detection_confidence_threshold": Config.DETECTION_CONFIDENCE_THRESHOLD,
            "api_config": Config.get_api_config()
        }
        
        return APIResponse(
            success=True,
            message="System information retrieved successfully",
            data=system_info
        )
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_DEBUG
    )
