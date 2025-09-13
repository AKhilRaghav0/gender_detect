"""
Database Setup Script for Bus Safety Gender Detection System
Automatically creates the required database and collections in Appwrite
"""

import logging
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.collections import Collections
from appwrite.services.attributes import Attributes
from appwrite.services.indexes import Indexes
from appwrite.exception import AppwriteException
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Class to handle database and collection setup"""
    
    def __init__(self):
        """Initialize Appwrite client"""
        self.client = Client()
        self.client.set_endpoint(Config.APPWRITE_PUBLIC_ENDPOINT)
        self.client.set_project(Config.APPWRITE_PROJECT_ID)
        self.client.set_key(Config.APPWRITE_API_KEY)
        
        self.databases = Databases(self.client)
        self.collections = Collections(self.client)
        self.attributes = Attributes(self.client)
        self.indexes = Indexes(self.client)
        
        logger.info("Database setup client initialized")
    
    def create_database(self) -> bool:
        """Create the main database"""
        try:
            logger.info(f"Creating database: {Config.DATABASE_ID}")
            
            # Check if database already exists
            try:
                existing_db = self.databases.get(Config.DATABASE_ID)
                logger.info(f"Database {Config.DATABASE_ID} already exists")
                return True
            except AppwriteException as e:
                if "Database not found" in str(e):
                    pass  # Database doesn't exist, create it
                else:
                    raise e
            
            # Create new database
            result = self.databases.create(
                database_id=Config.DATABASE_ID,
                name="Bus Safety Gender Detection System",
                enabled=True
            )
            
            logger.info(f"Database created successfully: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def create_buses_collection(self) -> bool:
        """Create the buses collection with proper attributes"""
        try:
            logger.info(f"Creating collection: {Config.BUS_COLLECTION_ID}")
            
            # Check if collection exists
            try:
                existing_collection = self.collections.get(
                    database_id=Config.DATABASE_ID,
                    collection_id=Config.BUS_COLLECTION_ID
                )
                logger.info(f"Collection {Config.BUS_COLLECTION_ID} already exists")
                return True
            except AppwriteException as e:
                if "Collection not found" in str(e):
                    pass  # Collection doesn't exist, create it
                else:
                    raise e
            
            # Create collection
            collection = self.collections.create(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                name="Buses",
                permissions=["read('any')", "write('any')"],
                document_security=False
            )
            
            logger.info(f"Collection created: {collection}")
            
            # Add attributes
            self._add_bus_attributes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create buses collection: {e}")
            return False
    
    def create_gender_counts_collection(self) -> bool:
        """Create the gender counts collection"""
        try:
            logger.info(f"Creating collection: {Config.GENDER_COUNT_COLLECTION_ID}")
            
            # Check if collection exists
            try:
                existing_collection = self.collections.get(
                    database_id=Config.DATABASE_ID,
                    collection_id=Config.GENDER_COUNT_COLLECTION_ID
                )
                logger.info(f"Collection {Config.GENDER_COUNT_COLLECTION_ID} already exists")
                return True
            except AppwriteException as e:
                if "Collection not found" in str(e):
                    pass
                else:
                    raise e
            
            # Create collection
            collection = self.collections.create(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                name="Gender Counts",
                permissions=["read('any')", "write('any')"],
                document_security=False
            )
            
            logger.info(f"Collection created: {collection}")
            
            # Add attributes
            self._add_gender_count_attributes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create gender counts collection: {e}")
            return False
    
    def create_routes_collection(self) -> bool:
        """Create the routes collection"""
        try:
            logger.info(f"Creating collection: {Config.ROUTE_COLLECTION_ID}")
            
            # Check if collection exists
            try:
                existing_collection = self.collections.get(
                    database_id=Config.DATABASE_ID,
                    collection_id=Config.ROUTE_COLLECTION_ID
                )
                logger.info(f"Collection {Config.ROUTE_COLLECTION_ID} already exists")
                return True
            except AppwriteException as e:
                if "Collection not found" in str(e):
                    pass
                else:
                    raise e
            
            # Create collection
            collection = self.collections.create(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                name="Routes",
                permissions=["read('any')", "write('any')"],
                document_security=False
            )
            
            logger.info(f"Collection created: {collection}")
            
            # Add attributes
            self._add_route_attributes()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create routes collection: {e}")
            return False
    
    def _add_bus_attributes(self):
        """Add attributes to buses collection"""
        try:
            # Bus ID (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="bus_id",
                size=50,
                required=True
            )
            
            # Route number (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="route_number",
                size=10,
                required=True
            )
            
            # Driver name (string, optional)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="driver_name",
                size=100,
                required=False
            )
            
            # Driver contact (string, optional)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="driver_contact",
                size=20,
                required=False
            )
            
            # Capacity (integer, required)
            self.attributes.create_integer(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="capacity",
                required=True,
                min=1,
                max=100
            )
            
            # Status (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="status",
                size=20,
                required=True
            )
            
            # Is active (boolean, required)
            self.attributes.create_boolean(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="is_active",
                required=True
            )
            
            # Last updated (datetime, required)
            self.attributes.create_datetime(
                database_id=Config.DATABASE_ID,
                collection_id=Config.BUS_COLLECTION_ID,
                key="last_updated",
                required=True
            )
            
            logger.info("Bus attributes added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add bus attributes: {e}")
    
    def _add_gender_count_attributes(self):
        """Add attributes to gender counts collection"""
        try:
            # Bus ID (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="bus_id",
                size=50,
                required=True
            )
            
            # Male count (integer, required)
            self.attributes.create_integer(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="male_count",
                required=True,
                min=0,
                max=100
            )
            
            # Female count (integer, required)
            self.attributes.create_integer(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="female_count",
                required=True,
                min=0,
                max=100
            )
            
            # Total count (integer, required)
            self.attributes.create_integer(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="total_count",
                required=True,
                min=0,
                max=100
            )
            
            # Confidence (float, required)
            self.attributes.create_float(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="confidence",
                required=True,
                min=0.0,
                max=1.0
            )
            
            # Timestamp (datetime, required)
            self.attributes.create_datetime(
                database_id=Config.DATABASE_ID,
                collection_id=Config.GENDER_COUNT_COLLECTION_ID,
                key="timestamp",
                required=True
            )
            
            logger.info("Gender count attributes added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add gender count attributes: {e}")
    
    def _add_route_attributes(self):
        """Add attributes to routes collection"""
        try:
            # Route number (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                key="route_number",
                size=10,
                required=True
            )
            
            # Route name (string, required)
            self.attributes.create_string(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                key="route_name",
                size=100,
                required=True
            )
            
            # Safety score (float, required)
            self.attributes.create_float(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                key="safety_score",
                required=True,
                min=0.0,
                max=1.0
            )
            
            # Is active (boolean, required)
            self.attributes.create_boolean(
                database_id=Config.DATABASE_ID,
                collection_id=Config.ROUTE_COLLECTION_ID,
                key="is_active",
                required=True
            )
            
            logger.info("Route attributes added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add route attributes: {e}")
    
    def setup_complete_database(self) -> bool:
        """Set up the complete database structure"""
        try:
            logger.info("Starting complete database setup...")
            
            # Create database
            if not self.create_database():
                logger.error("Failed to create database")
                return False
            
            # Create collections
            if not self.create_buses_collection():
                logger.error("Failed to create buses collection")
                return False
            
            if not self.create_gender_counts_collection():
                logger.error("Failed to create gender counts collection")
                return False
            
            if not self.create_routes_collection():
                logger.error("Failed to create routes collection")
                return False
            
            logger.info("✅ Complete database setup successful!")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def populate_default_routes(self):
        """Populate default route data"""
        try:
            logger.info("Populating default route data...")
            
            from models import RouteInfo
            
            for route_num, route_data in Config.GURUGRAM_ROUTES.items():
                try:
                    route_info = RouteInfo(
                        route_number=route_num,
                        route_name=route_data["name"],
                        safety_score=route_data["safety_score"],
                        stops=route_data["stops"],
                        alternative_routes=[r for r in Config.GURUGRAM_ROUTES.keys() if r != route_num],
                        is_active=True
                    )
                    
                    # Use model_dump() for newer Pydantic versions, fallback to dict()
                    try:
                        route_data_dict = route_info.model_dump()
                    except AttributeError:
                        route_data_dict = route_info.dict()
                    
                    # Create route document
                    self.databases.create_document(
                        database_id=Config.DATABASE_ID,
                        collection_id=Config.ROUTE_COLLECTION_ID,
                        document_id=route_num,
                        data=route_data_dict
                    )
                    
                    logger.info(f"Route {route_num} populated successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to populate route {route_num}: {e}")
            
            logger.info("Default routes populated successfully")
            
        except Exception as e:
            logger.error(f"Failed to populate default routes: {e}")

def main():
    """Main function to run database setup"""
    try:
        logger.info("🚀 Starting Appwrite Database Setup...")
        
        setup = DatabaseSetup()
        
        # Set up complete database structure
        if setup.setup_complete_database():
            logger.info("Database structure created successfully!")
            
            # Populate default routes
            setup.populate_default_routes()
            
            logger.info("🎉 Database setup completed successfully!")
            logger.info(f"Database ID: {Config.DATABASE_ID}")
            logger.info(f"Collections: {Config.BUS_COLLECTION_ID}, {Config.GENDER_COUNT_COLLECTION_ID}, {Config.ROUTE_COLLECTION_ID}")
            
        else:
            logger.error("❌ Database setup failed!")
            
    except Exception as e:
        logger.error(f"Setup script failed: {e}")

if __name__ == "__main__":
    main()
