# 🚌 Bus Saheli Backend - Low-Level Design

## 🏗️ Architecture Components

### 1. Core Services

#### Gender Detection Service
```python
class GenderDetectionService:
    def __init__(self):
        self.insightface_app = None
        self.haar_cascade = None
        self.face_tracker = FaceTracker()
    
    def detect_gender(self, image: np.ndarray) -> GenderResult:
        # Multi-algorithm ensemble detection
        pass
    
    def track_faces(self, faces: List[Face]) -> List[TrackedFace]:
        # Face tracking to prevent double counting
        pass
```

#### Safety Algorithm Service
```python
class SafetyAlgorithmService:
    def calculate_safety_score(self, bus_data: BusData) -> SafetyScore:
        # Female ratio: 40% weight
        # Capacity ratio: 30% weight  
        # Crowd density: 20% weight
        # Historical data: 10% weight
        pass
    
    def get_safety_recommendations(self, score: float) -> List[str]:
        # Generate safety recommendations
        pass
```

#### Redis Integration Service
```python
class RedisService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.stream_name = "bus_safety_stream"
    
    def publish_bus_data(self, bus_data: BusData):
        # Publish to Redis stream
        pass
    
    def get_active_routes(self) -> List[Route]:
        # Get active bus routes
        pass
```

### 2. API Endpoints

#### Real-time Data
- `POST /api/v1/detect-gender` - Process camera frame
- `GET /api/v1/buses/{bus_id}/live-data` - Get live bus data
- `WebSocket /ws/bus-safety` - Real-time updates

#### Route Management
- `GET /api/v1/routes` - List all routes
- `GET /api/v1/routes/{route_id}/buses` - Get buses on route
- `POST /api/v1/routes/{route_id}/register-bus` - Register new bus

#### Safety Analytics
- `GET /api/v1/safety-metrics` - Overall safety metrics
- `GET /api/v1/routes/{route_id}/safety-history` - Historical safety data
- `GET /api/v1/alerts` - Safety alerts and notifications

### 3. Data Models

#### Bus Data Model
```python
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
    last_updated: datetime
    is_active: bool
    location: Optional[Location] = None
```

#### Safety Metrics Model
```python
@dataclass
class SafetyMetrics:
    route_id: str
    safety_level: SafetyLevel
    female_ratio: float
    capacity_ratio: float
    crowd_density: CrowdDensity
    safety_score: float
    recommendations: List[str]
    timestamp: datetime
```

### 4. Database Schema

#### Redis Keys Structure
```
bus_safety:bus:{bus_id} -> BusData (JSON)
bus_safety:route:{route_id}:active -> Set of active bus IDs
bus_safety:stream -> Redis Stream for real-time data
bus_safety:alerts -> List of safety alerts
```

#### PostgreSQL Tables
```sql
-- Buses table
CREATE TABLE buses (
    id SERIAL PRIMARY KEY,
    bus_id VARCHAR(50) UNIQUE NOT NULL,
    route_number VARCHAR(10) NOT NULL,
    capacity INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Safety metrics table
CREATE TABLE safety_metrics (
    id SERIAL PRIMARY KEY,
    bus_id VARCHAR(50) NOT NULL,
    route_number VARCHAR(10) NOT NULL,
    female_count INTEGER NOT NULL,
    male_count INTEGER NOT NULL,
    total_count INTEGER NOT NULL,
    safety_score FLOAT NOT NULL,
    safety_level VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Routes table
CREATE TABLE routes (
    id SERIAL PRIMARY KEY,
    route_number VARCHAR(10) UNIQUE NOT NULL,
    route_name VARCHAR(100) NOT NULL,
    stops JSONB NOT NULL,
    safety_threshold FLOAT DEFAULT 0.4,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔧 Implementation Plan

### Phase 1: Core Backend (Week 1-2)
1. **Setup project structure**
2. **Implement Redis integration**
3. **Create basic API endpoints**
4. **Implement safety algorithm**
5. **Add face tracking system**

### Phase 2: Advanced Features (Week 3-4)
1. **WebSocket implementation**
2. **Real-time data streaming**
3. **Safety alerts system**
4. **Performance optimization**
5. **Error handling and logging**

### Phase 3: Integration (Week 5-6)
1. **Flutter app API integration**
2. **Admin panel development**
3. **Testing and debugging**
4. **Deployment preparation**

## 🚀 Quick Start

### Prerequisites
```bash
# Install Redis
sudo apt-get install redis-server

# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Install Python dependencies
pip install -r requirements.txt
```

### Setup
```bash
# Start Redis
redis-server

# Start PostgreSQL
sudo systemctl start postgresql

# Run migrations
python manage.py migrate

# Start the server
python main.py
```

## 📊 Performance Targets

- **Detection Speed**: < 100ms per frame
- **API Response**: < 200ms for all endpoints
- **WebSocket Latency**: < 50ms for real-time updates
- **Memory Usage**: < 512MB per Raspberry Pi
- **Uptime**: 99.9% availability

## 🔍 Monitoring

- **Health Checks**: `/health` endpoint
- **Metrics**: Prometheus integration
- **Logging**: Structured logging with ELK stack
- **Alerts**: Slack/Email notifications for critical issues
