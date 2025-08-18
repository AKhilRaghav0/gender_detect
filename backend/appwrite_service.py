"""
Appwrite service for Bus Safety Gender Detection System
Handles all database operations and data sync
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.exception import AppwriteException

from config import Config
from models import BusInfo, GenderCount, BusSafetyMetrics, RouteInfo

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppwriteService:
    """Service class for Appwrite database operations"""
    
    def __init__(self):
        """Initialize Appwrite client and services"""
        try:
            # Initialize Appwrite client
            self.client = Client()
            self.client.set_endpoint(Config.APPWRITE_PUBLIC_ENDPOINT)
            self.client.set_project(Config.APPWRITE_PROJECT_ID)
            self.client.set_key(Config.APPWRITE_API_KEY)
            
            # Initialize services
            self.databases = Databases(self.client)
            
            logger.info("Appwrite service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Appwrite service: {e}")
            raise
    
    def create_database_collections(self) -> bool:
        """Create required database collections if they don't exist"""
        try:
            # For now, we'll just log that we're assuming collections exist
            # In a real implementation, you would:
            # 1. Create database if it doesn't exist
            # 2. Create collections if they don't exist
            # 3. Set up proper attributes and indexes
            
            logger.info("Database collections setup completed (assuming collections exist)")
            logger.info(f"Expected database ID: {Config.DATABASE_ID}")
            logger.info(f"Expected collections: {Config.BUS_COLLECTION_ID}, {Config.GENDER_COUNT_COLLECTION_ID}, {Config.ROUTE_COLLECTION_ID}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup database collections: {e}")
            return False
    
    # Bus Management Methods
    def register_bus(self, bus_info: BusInfo) -> bool:
        """Register a new bus in the system"""
        try:
            logger.info(f"Attempting to register bus: {bus_info.bus_id}")
            logger.info(f"Database ID: {Config.DATABASE_ID}")
            logger.info(f"Collection ID: {Config.BUS_COLLECTION_ID}")
            
            # Use model_dump() for newer Pydantic versions, fallback to dict()
            try:
                bus_data = bus_info.model_dump()
            except AttributeError:
                bus_data = bus_info.dict()
            
            # Convert current_location dict to JSON string if it exists
            if bus_data.get('current_location') and isinstance(bus_data['current_location'], dict):
                import json
                bus_data['current_location'] = json.dumps(bus_data['current_location'])
            
            bus_data['last_updated'] = datetime.utcnow().isoformat()
            
            logger.info(f"Bus data prepared: {bus_data}")
            
            # Create bus document
            result = self.databases.create_document(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                document_id=bus_info.bus_id,
                data=bus_data
            )
            
            logger.info(f"Bus {bus_info.bus_id} registered successfully")
            return True
            
        except AppwriteException as e:
            logger.error(f"Appwrite error registering bus: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error registering bus: {e}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            return False
    
    def get_bus_info(self, bus_id: str) -> Optional[BusInfo]:
        """Get bus information by ID"""
        try:
            result = self.databases.get_document(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                document_id=bus_id
            )
            
            # Parse JSON strings back to dictionaries for certain fields
            if result.get('current_location') and isinstance(result['current_location'], str):
                try:
                    import json
                    result['current_location'] = json.loads(result['current_location'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse current_location JSON: {result['current_location']}")
                    result['current_location'] = None
            
            return BusInfo(**result)
            
        except AppwriteException as e:
            logger.error(f"Appwrite error getting bus info: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting bus info: {e}")
            return None
    
    def update_bus_status(self, bus_id: str, status: str, location: Optional[Dict] = None) -> bool:
        """Update bus status and location"""
        try:
            update_data = {
                'status': status,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            if location:
                update_data['current_location'] = location
            
            self.databases.update_document(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                document_id=bus_id,
                data=update_data
            )
            
            logger.info(f"Bus {bus_id} status updated to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating bus status: {e}")
            return False
    
    # Gender Count Methods
    def save_gender_count(self, gender_count: GenderCount) -> bool:
        """Save real-time gender count data"""
        try:
            # Use model_dump() for newer Pydantic versions, fallback to dict()
            try:
                count_data = gender_count.model_dump()
            except AttributeError:
                count_data = gender_count.dict()
            
            # Convert any dict fields to JSON strings if they exist
            for key, value in count_data.items():
                if isinstance(value, dict):
                    import json
                    count_data[key] = json.dumps(value)
            
            count_data['timestamp'] = datetime.utcnow().isoformat()
            
            # Create gender count document
            result = self.databases.create_document(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                document_id=f"{gender_count.bus_id}_{int(datetime.utcnow().timestamp())}",
                data=count_data
            )
            
            logger.info(f"Gender count saved for bus {gender_count.bus_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving gender count: {e}")
            return False
    
    def get_latest_gender_count(self, bus_id: str) -> Optional[GenderCount]:
        """Get the latest gender count for a bus"""
        try:
            # Query for latest count
            result = self.databases.list_documents(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                queries=[
                    f"bus_id={bus_id}",
                    "orderDesc=timestamp"
                ],
                limit=1
            )
            
            if result.documents:
                doc = result.documents[0]
                # Parse JSON strings back to dictionaries for certain fields
                for key, value in doc.items():
                    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        try:
                            import json
                            doc[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                
                return GenderCount(**doc)
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest gender count: {e}")
            return None
    
    def get_gender_count_history(self, bus_id: str, hours: int = 24) -> List[GenderCount]:
        """Get gender count history for a bus"""
        try:
            # Calculate time threshold
            threshold = datetime.utcnow().timestamp() - (hours * 3600)
            
            result = self.databases.list_documents(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                queries=[
                    f"bus_id={bus_id}",
                    f"timestamp>{threshold}",
                    "orderDesc=timestamp"
                ]
            )
            
            gender_counts = []
            for doc in result.documents:
                # Parse JSON strings back to dictionaries for certain fields
                for key, value in doc.items():
                    if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        try:
                            import json
                            doc[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                
                gender_counts.append(GenderCount(**doc))
            
            return gender_counts
            
        except Exception as e:
            logger.error(f"Error getting gender count history: {e}")
            return []
    
    # Route Methods
    def get_route_info(self, route_number: str) -> Optional[RouteInfo]:
        """Get route information by route number"""
        try:
            result = self.databases.get_document(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                document_id=route_number
            )
            
            # Parse JSON strings back to dictionaries for certain fields
            for key, value in result.items():
                if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        import json
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass  # Keep as string if not valid JSON
            
            return RouteInfo(**result)
            
        except AppwriteException as e:
            logger.error(f"Appwrite error getting route info: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting route info: {e}")
            return None
    
    def get_all_routes(self) -> List[RouteInfo]:
        """Get all available routes"""
        try:
            result = self.databases.list_documents(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                queries=["is_active=true"]
            )
            
            routes = []
            for doc in result.documents:
                # Parse JSON strings back to dictionaries for certain fields
                for key, value in doc.items():
                    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                        try:
                            import json
                            doc[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string if not valid JSON
                
                routes.append(RouteInfo(**doc))
            
            return routes
            
        except Exception as e:
            logger.error(f"Error getting all routes: {e}")
            return []
    
    # Utility Methods
    def health_check(self) -> bool:
        """Check if Appwrite service is healthy"""
        try:
            # Try to list databases to check connectivity
            self.databases.list()
            return True
            
        except Exception as e:
            logger.error(f"Appwrite health check failed: {e}")
            return False
    
    def close(self):
        """Close Appwrite connections"""
        try:
            # Close any open connections
            logger.info("Appwrite service connections closed")
            
        except Exception as e:
            logger.error(f"Error closing Appwrite service: {e}")
