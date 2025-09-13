# 🚌 Bus Saheli - Women's Safety in Public Transport

## 🎯 Project Overview
Real-time gender detection and safety monitoring system for buses in Gurugram, providing women with safety information before boarding.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │   Backend API    │    │   Redis DB      │    │   Flutter App   │
│   (Camera)      │───▶│   (FastAPI)      │───▶│   (Real-time)   │◀───│   (Mobile)      │
│   Route: 118    │    │   + Safety Algo  │    │   + Analytics   │    │   + UI/UX       │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Admin Panel    │
                       │   (Web)          │
                       └──────────────────┘
```

## 🚀 Core Features

### 1. Real-time Gender Detection
- **Face Detection**: InsightFace + OpenCV + Haar Cascade fallback
- **Gender Classification**: Multi-algorithm ensemble for accuracy
- **Face Tracking**: Prevents double counting
- **Performance**: 30+ FPS on Raspberry Pi 5

### 2. Safety Algorithm
- **Female Ratio**: Percentage of female passengers
- **Capacity Analysis**: Total vs. occupied seats
- **Crowd Density**: Safety based on passenger density
- **Route Safety Score**: Historical safety data per route

### 3. Real-time Data Flow
- **Redis Streams**: Real-time data ingestion
- **WebSocket**: Live updates to Flutter app
- **REST APIs**: Historical data and analytics
- **Data Persistence**: PostgreSQL for long-term storage

### 4. Flutter Mobile App
- **Route Selection**: List of active bus routes
- **Live Safety Data**: Real-time passenger counts
- **Safety Alerts**: Notifications for unsafe conditions
- **Historical Data**: Past safety trends

## 📱 Flutter App Features

### Screens
1. **Home Screen**: Active routes with safety indicators
2. **Route Details**: Live passenger count and safety score
3. **Safety Alerts**: Push notifications for safety concerns
4. **Settings**: User preferences and notifications
5. **About**: App information and contact details

### Safety Indicators
- 🟢 **Safe**: >40% female passengers, normal capacity
- 🟡 **Moderate**: 20-40% female passengers, high capacity
- 🔴 **Unsafe**: <20% female passengers, overcrowded

## 🔧 Technical Stack

### Backend
- **FastAPI**: REST API framework
- **Redis**: Real-time data storage and caching
- **PostgreSQL**: Persistent data storage
- **WebSocket**: Real-time communication
- **OpenCV + InsightFace**: Computer vision

### Mobile App
- **Flutter**: Cross-platform mobile development
- **Provider**: State management
- **HTTP**: API communication
- **WebSocket**: Real-time updates
- **Local Storage**: Offline data caching

### Infrastructure
- **Docker**: Containerization
- **Nginx**: Reverse proxy
- **SSL/TLS**: Secure communication
- **Monitoring**: System health and performance

## 📊 Data Models

### Bus Data
```json
{
  "bus_id": "BUS_118_001",
  "route_number": "118",
  "current_stop": "Gurgaon Metro",
  "capacity": 50,
  "passenger_count": 35,
  "female_count": 15,
  "male_count": 20,
  "safety_score": 8.5,
  "last_updated": "2024-01-15T10:30:00Z",
  "is_active": true
}
```

### Safety Metrics
```json
{
  "route_id": "118",
  "safety_level": "SAFE",
  "female_ratio": 0.43,
  "capacity_ratio": 0.70,
  "crowd_density": "NORMAL",
  "safety_score": 8.5,
  "recommendations": ["Safe to travel", "Well-lit areas"]
}
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Flutter SDK
- Redis Server
- PostgreSQL
- Docker (optional)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Flutter App Setup
```bash
cd BusSaheli/flutter_app
flutter pub get
flutter run
```

## 📈 Future Enhancements

1. **AI Safety Predictions**: ML models for safety forecasting
2. **Emergency Features**: Panic button and emergency contacts
3. **Social Features**: User reviews and safety reports
4. **Integration**: GPS tracking and route optimization
5. **Analytics**: Advanced safety analytics and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
