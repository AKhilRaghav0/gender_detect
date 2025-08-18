"""
Simple Database Setup Script for Bus Safety Gender Detection System
Works with Appwrite version 11.1.0
"""

import logging
import requests
import json
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDatabaseSetup:
    """Simple database setup using direct HTTP requests"""
    
    def __init__(self):
        self.base_url = Config.APPWRITE_PUBLIC_ENDPOINT
        self.project_id = Config.APPWRITE_PROJECT_ID
        self.api_key = Config.APPWRITE_API_KEY
        
        self.headers = {
            'X-Appwrite-Project': self.project_id,
            'X-Appwrite-Key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        logger.info("Simple database setup initialized")
    
    def create_database(self):
        """Create the main database using HTTP API"""
        try:
            url = f"{self.base_url}/databases"
            
            data = {
                "databaseId": Config.DATABASE_ID,
                "name": "Bus Safety Gender Detection System"
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                logger.info(f"✅ Database '{Config.DATABASE_ID}' created successfully!")
                return True
            elif response.status_code == 409:
                logger.info(f"✅ Database '{Config.DATABASE_ID}' already exists!")
                return True
            else:
                logger.error(f"❌ Failed to create database: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating database: {e}")
            return False
    
    def create_collection(self, collection_id, name, attributes):
        """Create a collection with attributes"""
        try:
            url = f"{self.base_url}/databases/{Config.DATABASE_ID}/collections"
            
            data = {
                "collectionId": collection_id,
                "name": name
                # Removed permissions temporarily to test
            }
            
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                logger.info(f"✅ Collection '{collection_id}' created successfully!")
                
                # Add attributes
                self._add_attributes(collection_id, attributes)
                return True
                
            elif response.status_code == 409:
                logger.info(f"✅ Collection '{collection_id}' already exists!")
                return True
            else:
                logger.error(f"❌ Failed to create collection '{collection_id}': {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating collection '{collection_id}': {e}")
            return False
    
    def _add_attributes(self, collection_id, attributes):
        """Add attributes to a collection"""
        try:
            for attr in attributes:
                url = f"{self.base_url}/databases/{Config.DATABASE_ID}/collections/{collection_id}/attributes/{attr['type']}"
                
                data = {
                    "key": attr['key'],
                    "required": attr.get('required', False),
                    "size": attr.get('size', 255),
                    "min": attr.get('min'),
                    "max": attr.get('max')
                }
                
                # Remove None values
                data = {k: v for k, v in data.items() if v is not None}
                
                response = requests.post(url, headers=self.headers, json=data)
                
                if response.status_code in [200, 201]:
                    logger.info(f"  ✅ Attribute '{attr['key']}' added to '{collection_id}'")
                else:
                    logger.warning(f"  ⚠️ Attribute '{attr['key']}' may already exist: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Error adding attributes to '{collection_id}': {e}")
    
    def create_buses_collection(self):
        """Create the buses collection"""
        attributes = [
            {"type": "string", "key": "bus_id", "required": True, "size": 50},
            {"type": "string", "key": "route_number", "required": True, "size": 10},
            {"type": "string", "key": "driver_name", "required": False, "size": 100},
            {"type": "string", "key": "driver_contact", "required": False, "size": 20},
            {"type": "integer", "key": "capacity", "required": True, "min": 1, "max": 100},
            {"type": "string", "key": "current_location", "required": False, "size": 255},  # Added missing field
            {"type": "string", "key": "status", "required": True, "size": 20},
            {"type": "boolean", "key": "is_active", "required": True},
            {"type": "datetime", "key": "last_updated", "required": True}
        ]
        
        return self.create_collection(Config.BUS_COLLECTION_ID, "Buses", attributes)
    
    def create_gender_counts_collection(self):
        """Create the gender counts collection"""
        attributes = [
            {"type": "string", "key": "bus_id", "required": True, "size": 50},
            {"type": "integer", "key": "male_count", "required": True, "min": 0, "max": 100},
            {"type": "integer", "key": "female_count", "required": True, "min": 0, "max": 100},
            {"type": "integer", "key": "total_count", "required": True, "min": 0, "max": 100},
            {"type": "float", "key": "confidence", "required": True, "min": 0.0, "max": 1.0},
            {"type": "datetime", "key": "timestamp", "required": True}
        ]
        
        return self.create_collection(Config.GENDER_COUNT_COLLECTION_ID, "Gender Counts", attributes)
    
    def create_routes_collection(self):
        """Create the routes collection"""
        attributes = [
            {"type": "string", "key": "route_number", "required": True, "size": 10},
            {"type": "string", "key": "route_name", "required": True, "size": 100},
            {"type": "float", "key": "safety_score", "required": True, "min": 0.0, "max": 1.0},
            {"type": "boolean", "key": "is_active", "required": True}
        ]
        
        return self.create_collection(Config.ROUTE_COLLECTION_ID, "Routes", attributes)
    
    def populate_default_routes(self):
        """Populate default route data"""
        try:
            logger.info("🚀 Populating default route data...")
            
            for route_num, route_data in Config.GURUGRAM_ROUTES.items():
                try:
                    url = f"{self.base_url}/databases/{Config.DATABASE_ID}/collections/{Config.ROUTE_COLLECTION_ID}/documents"
                    
                    data = {
                        "route_number": route_num,
                        "route_name": route_data["name"],
                        "safety_score": route_data["safety_score"],
                        "stops": route_data["stops"],
                        "alternative_routes": [r for r in Config.GURUGRAM_ROUTES.keys() if r != route_num],
                        "is_active": True
                    }
                    
                    response = requests.post(url, headers=self.headers, json=data)
                    
                    if response.status_code == 201:
                        logger.info(f"  ✅ Route {route_num} populated successfully")
                    else:
                        logger.warning(f"  ⚠️ Route {route_num} may already exist: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"  ❌ Failed to populate route {route_num}: {e}")
            
            logger.info("✅ Default routes populated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to populate default routes: {e}")
    
    def setup_complete_database(self):
        """Set up the complete database structure"""
        try:
            logger.info("🚀 Starting complete database setup...")
            
            # Create database
            if not self.create_database():
                logger.error("❌ Failed to create database")
                return False
            
            # Create collections
            if not self.create_buses_collection():
                logger.error("❌ Failed to create buses collection")
                return False
            
            if not self.create_gender_counts_collection():
                logger.error("❌ Failed to create gender counts collection")
                return False
            
            if not self.create_routes_collection():
                logger.error("❌ Failed to create routes collection")
                return False
            
            logger.info("✅ Complete database setup successful!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False

def main():
    """Main function to run database setup"""
    try:
        logger.info("🚀 Starting Simple Appwrite Database Setup...")
        logger.info(f"📡 Endpoint: {Config.APPWRITE_PUBLIC_ENDPOINT}")
        logger.info(f"🏢 Project: {Config.APPWRITE_PROJECT_ID}")
        
        setup = SimpleDatabaseSetup()
        
        # Set up complete database structure
        if setup.setup_complete_database():
            logger.info("🎉 Database structure created successfully!")
            
            # Populate default routes
            setup.populate_default_routes()
            
            logger.info("🎉 Database setup completed successfully!")
            logger.info(f"📊 Database ID: {Config.DATABASE_ID}")
            logger.info(f"📁 Collections: {Config.BUS_COLLECTION_ID}, {Config.GENDER_COUNT_COLLECTION_ID}, {Config.ROUTE_COLLECTION_ID}")
            
        else:
            logger.error("❌ Database setup failed!")
            
    except Exception as e:
        logger.error(f"❌ Setup script failed: {e}")

if __name__ == "__main__":
    main()
