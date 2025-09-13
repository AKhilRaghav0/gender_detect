#!/usr/bin/env python3
"""
Redis Integration for Bus Saheli
Real-time data storage and streaming for bus safety monitoring
"""

import redis
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    SAFE = "SAFE"
    MODERATE = "MODERATE"
    UNSAFE = "UNSAFE"

@dataclass
class BusData:
    bus_id: str
    route_number: str
    current_stop: str
    capacity: int
    passenger_count: int
    female_count: int
    male_count: int
    safety_score: float
    safety_level: SafetyLevel
    last_updated: datetime
    is_active: bool
    location: Optional[Dict[str, float]] = None

@dataclass
class SafetyMetrics:
    route_id: str
    safety_level: SafetyLevel
    female_ratio: float
    capacity_ratio: float
    crowd_density: str
    safety_score: float
    recommendations: List[str]
    timestamp: datetime

class RedisService:
    """Redis service for real-time data management"""
    
    def __init__(self, host='redis-16563.crce206.ap-south-1-1.ec2.redns.redis-cloud.com', 
                 port=16563, db=0, username="default", 
                 password="4D9I08GiJXxrFFQkVvLU5XV3hhv1mNMK"):
        self.redis_client = redis.Redis(
            host=host, 
            port=port, 
            db=db, 
            decode_responses=True,
            username=username,
            password=password
        )
        self.stream_name = "bus_safety_stream"
        self.bus_data_prefix = "bus_safety:bus:"
        self.route_prefix = "bus_safety:route:"
        self.alerts_key = "bus_safety:alerts"
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    def publish_bus_data(self, bus_data: BusData) -> bool:
        """Publish bus data to Redis stream"""
        try:
            # Convert to JSON-serializable format
            data = asdict(bus_data)
            data['last_updated'] = bus_data.last_updated.isoformat()
            data['safety_level'] = bus_data.safety_level.value
            data['location'] = bus_data.location or {}
            
            # Publish to stream
            message_id = self.redis_client.xadd(
                self.stream_name,
                data,
                maxlen=10000  # Keep last 10k messages
            )
            
            # Store current bus data
            bus_key = f"{self.bus_data_prefix}{bus_data.bus_id}"
            self.redis_client.setex(
                bus_key,
                timedelta(minutes=5),  # Expire after 5 minutes
                json.dumps(data)
            )
            
            # Update route active buses
            route_key = f"{self.route_prefix}{bus_data.route_number}:active"
            if bus_data.is_active:
                self.redis_client.sadd(route_key, bus_data.bus_id)
            else:
                self.redis_client.srem(route_key, bus_data.bus_id)
            
            # Set expiration for route key
            self.redis_client.expire(route_key, timedelta(minutes=10))
            
            logger.info(f"📡 Published bus data: {bus_data.bus_id} -> {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish bus data: {e}")
            return False
    
    def get_bus_data(self, bus_id: str) -> Optional[BusData]:
        """Get current bus data from Redis"""
        try:
            bus_key = f"{self.bus_data_prefix}{bus_id}"
            data = self.redis_client.get(bus_key)
            
            if not data:
                return None
            
            data_dict = json.loads(data)
            
            # Convert back to BusData object
            return BusData(
                bus_id=data_dict['bus_id'],
                route_number=data_dict['route_number'],
                current_stop=data_dict['current_stop'],
                capacity=data_dict['capacity'],
                passenger_count=data_dict['passenger_count'],
                female_count=data_dict['female_count'],
                male_count=data_dict['male_count'],
                safety_score=data_dict['safety_score'],
                safety_level=SafetyLevel(data_dict['safety_level']),
                last_updated=datetime.fromisoformat(data_dict['last_updated']),
                is_active=data_dict['is_active'],
                location=data_dict.get('location')
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get bus data: {e}")
            return None
    
    def get_active_routes(self) -> List[str]:
        """Get list of active route numbers"""
        try:
            # Get all route keys
            route_keys = self.redis_client.keys(f"{self.route_prefix}*:active")
            routes = []
            
            for key in route_keys:
                # Extract route number from key
                route_number = key.split(':')[-2]
                if self.redis_client.scard(key) > 0:  # Has active buses
                    routes.append(route_number)
            
            return routes
            
        except Exception as e:
            logger.error(f"❌ Failed to get active routes: {e}")
            return []
    
    def get_route_buses(self, route_number: str) -> List[BusData]:
        """Get all buses for a specific route"""
        try:
            route_key = f"{self.route_prefix}{route_number}:active"
            bus_ids = self.redis_client.smembers(route_key)
            
            buses = []
            for bus_id in bus_ids:
                bus_data = self.get_bus_data(bus_id)
                if bus_data:
                    buses.append(bus_data)
            
            return buses
            
        except Exception as e:
            logger.error(f"❌ Failed to get route buses: {e}")
            return []
    
    def add_safety_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Add safety alert to Redis"""
        try:
            alert_data['timestamp'] = datetime.now().isoformat()
            self.redis_client.lpush(self.alerts_key, json.dumps(alert_data))
            self.redis_client.ltrim(self.alerts_key, 0, 99)  # Keep last 100 alerts
            return True
        except Exception as e:
            logger.error(f"❌ Failed to add safety alert: {e}")
            return False
    
    def get_safety_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent safety alerts"""
        try:
            alerts = self.redis_client.lrange(self.alerts_key, 0, limit - 1)
            return [json.loads(alert) for alert in alerts]
        except Exception as e:
            logger.error(f"❌ Failed to get safety alerts: {e}")
            return []
    
    def subscribe_to_bus_updates(self, bus_id: str):
        """Subscribe to real-time bus updates"""
        try:
            # Create a consumer group for this bus
            group_name = f"bus_{bus_id}_group"
            consumer_name = f"consumer_{bus_id}_{datetime.now().timestamp()}"
            
            # Create consumer group if it doesn't exist
            try:
                self.redis_client.xgroup_create(self.stream_name, group_name, id='0', mkstream=True)
            except redis.ResponseError:
                pass  # Group already exists
            
            # Read from stream
            while True:
                messages = self.redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    {self.stream_name: '>'},
                    count=1,
                    block=1000  # 1 second timeout
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        if fields.get('bus_id') == bus_id:
                            yield fields
                            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to bus updates: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data from Redis"""
        try:
            # Remove expired bus data
            bus_keys = self.redis_client.keys(f"{self.bus_data_prefix}*")
            for key in bus_keys:
                if not self.redis_client.exists(key):
                    self.redis_client.delete(key)
            
            # Remove expired route data
            route_keys = self.redis_client.keys(f"{self.route_prefix}*:active")
            for key in route_keys:
                if not self.redis_client.exists(key):
                    self.redis_client.delete(key)
            
            logger.info("🧹 Cleaned up old Redis data")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get Redis system statistics"""
        try:
            info = self.redis_client.info()
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'stream_length': self.redis_client.xlen(self.stream_name),
                'active_routes': len(self.get_active_routes()),
                'total_buses': len(self.redis_client.keys(f"{self.bus_data_prefix}*"))
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system stats: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize Redis service
    redis_service = RedisService()
    
    # Create sample bus data
    sample_bus = BusData(
        bus_id="BUS_118_001",
        route_number="118",
        current_stop="Gurgaon Metro",
        capacity=50,
        passenger_count=35,
        female_count=15,
        male_count=20,
        safety_score=0.75,
        safety_level=SafetyLevel.SAFE,
        last_updated=datetime.now(),
        is_active=True,
        location={"lat": 28.4595, "lng": 77.0266}
    )
    
    # Publish data
    redis_service.publish_bus_data(sample_bus)
    
    # Get data back
    retrieved_bus = redis_service.get_bus_data("BUS_118_001")
    print(f"Retrieved bus: {retrieved_bus}")
    
    # Get active routes
    active_routes = redis_service.get_active_routes()
    print(f"Active routes: {active_routes}")
    
    # Get system stats
    stats = redis_service.get_system_stats()
    print(f"System stats: {stats}")
