# 🚌 Bus Safety Gender Detection System - Backend

## 📋 Overview

This is the backend system for the **Bus Safety Gender Detection System**, designed to provide real-time gender counting and safety monitoring for buses in Gurugram. The system integrates with your existing polished gender detection algorithm and provides a comprehensive REST API for bus management, real-time gender counting, and safety metrics.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raspberry Pi  │    │   Backend API    │    │   Appwrite DB   │
│   (Camera)      │───▶│   (FastAPI)      │───▶│   (Real-time)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Flutter App    │
                       │   (Frontend)     │
                       └──────────────────┘
```

## 🚀 Features

### Core Functionality
- **Real-time Gender Detection**: Integrates with your polished detection system
- **Bus Management**: Register, track, and manage bus information
- **Safety Metrics**: Calculate safety scores based on female ratio and capacity
- **Route Management**: Support for Gurugram routes (118, 111, 218)
- **Real-time Data Sync**: Appwrite integration for live updates

### API Endpoints
- **Health Check**: `/health` - System status monitoring
- **Gender Detection**: `/detect-gender` - Process images for gender counting
- **Bus Management**: `/buses/*` - Bus registration and status updates
- **Safety Metrics**: `/buses/{id}/safety-metrics` - Real-time safety calculations
- **Route Information**: `/routes/*` - Route details and alternatives

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- OpenCV with face detection cascades
- Appwrite account and project

### Installation

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Appwrite**:
   - Update `config.py` with your Appwrite credentials
   - Ensure your Appwrite project has the required collections

4. **Start the backend server**:
   ```bash
   python main.py
   ```

   The server will start on `http://localhost:8000`

### Environment Configuration

The system uses the following configuration (in `config.py`):
- **Appwrite Settings**: Project ID, endpoint, and API key
- **API Settings**: Host, port, and debug mode
- **Detection Settings**: Confidence thresholds and face detection parameters
- **Safety Settings**: Female ratio thresholds and safety scoring weights
- **Route Information**: Gurugram bus routes with safety scores

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: `/docs` - Interactive API documentation
- **ReDoc**: `/redoc` - Alternative API documentation

### Key Endpoints

#### 1. Health Check
```http
GET /health
```
Returns system health status including Appwrite and detection service status.

#### 2. Gender Detection
```http
POST /detect-gender
Content-Type: application/json

{
  "bus_id": "BUS_001",
  "image_data": "base64_encoded_image_string",
  "timestamp": null
}
```
Processes an image and returns gender counts with confidence scores.

#### 3. Bus Registration
```http
POST /buses/register
Content-Type: application/json

{
  "bus_id": "BUS_001",
  "route_number": "118",
  "driver_name": "Rajesh Kumar",
  "capacity": 50,
  "current_location": {"lat": 28.4595, "lng": 77.0266}
}
```

#### 4. Safety Metrics
```http
GET /buses/{bus_id}/safety-metrics
```
Returns comprehensive safety metrics including:
- Safety level (Safe/Moderate/Unsafe)
- Female passenger ratio
- Capacity utilization
- Route safety score
- Safety recommendations

## 🔧 Testing

### Run Backend Tests
```bash
cd backend
python test_backend.py
```

This will test all major API endpoints and verify the system is working correctly.

### Manual Testing
1. Start the backend server
2. Open `http://localhost:8000/docs` in your browser
3. Use the interactive Swagger UI to test endpoints
4. Monitor logs in the terminal for debugging information

## 📊 Data Models

### Bus Information
- **Bus ID**: Unique identifier
- **Route Number**: Bus route (118, 111, 218)
- **Driver Details**: Name and contact information
- **Capacity**: Maximum passenger capacity
- **Location**: GPS coordinates
- **Status**: Active/Inactive/Maintenance/Offline

### Gender Counts
- **Male Count**: Number of male passengers
- **Female Count**: Number of female passengers
- **Total Passengers**: Combined count
- **Confidence Score**: Detection accuracy
- **Timestamp**: When the count was taken

### Safety Metrics
- **Safety Level**: Safe/Moderate/Unsafe classification
- **Safety Score**: 0.0-1.0 overall safety rating
- **Female Ratio**: Percentage of female passengers
- **Capacity Utilization**: Bus fullness percentage
- **Recommendations**: Safety suggestions and alternatives

## 🔐 Security & Configuration

### CORS Settings
Currently configured to allow all origins for development. Configure appropriately for production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Production domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Appwrite Security
- Uses API key authentication
- Implements proper error handling
- Real-time subscriptions for live updates

## 🚨 Troubleshooting

### Common Issues

1. **Appwrite Connection Failed**
   - Verify your Appwrite credentials in `config.py`
   - Check if your Appwrite instance is running
   - Ensure your API key has proper permissions

2. **OpenCV Cascade Loading Failed**
   - Verify OpenCV installation: `pip install opencv-python`
   - Check if cascade files exist in OpenCV data directory

3. **Port Already in Use**
   - Change port in `config.py`: `API_PORT = 8001`
   - Kill existing process: `lsof -ti:8000 | xargs kill -9`

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and virtual environment

### Logs
- **Application Logs**: Check terminal output
- **File Logs**: Check `backend.log` file
- **Log Level**: Configure in `config.py` (DEBUG, INFO, WARNING, ERROR)

## 🔄 Integration with Raspberry Pi

### For Production Deployment
1. **Update Configuration**: Change `API_HOST` to `0.0.0.0` for external access
2. **Network Configuration**: Ensure Pi can reach your Appwrite server
3. **Camera Integration**: Use Pi Camera module with the gender detection service
4. **Performance Optimization**: Adjust detection parameters for Pi's CPU capabilities

### Real-time Data Flow
```
Pi Camera → Image Capture → Gender Detection → API Call → Appwrite DB → Flutter App
```

## 📈 Performance & Scaling

### Current Optimizations
- **Background Tasks**: Database operations run asynchronously
- **Efficient Image Processing**: Optimized OpenCV operations
- **Connection Pooling**: Appwrite client connection management

### Future Improvements
- **Caching**: Redis for frequently accessed data
- **Load Balancing**: Multiple backend instances
- **Database Optimization**: Indexing and query optimization
- **GPU Acceleration**: CUDA support for faster detection

## 🤝 Contributing

### Development Workflow
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and test thoroughly
3. Update documentation and tests
4. Submit pull request

### Code Standards
- Follow PEP 8 Python style guide
- Add type hints for all functions
- Include docstrings for all classes and methods
- Write tests for new functionality

## 📞 Support

### Getting Help
- Check the troubleshooting section above
- Review API documentation at `/docs`
- Check logs for error details
- Verify configuration settings

### Contact
- **Developer**: Akhil
- **Project Manager**: AI Assistant
- **System**: Bus Safety Gender Detection System

---

## 🎯 Next Steps

1. **Test the backend** with the provided test script
2. **Configure Appwrite** collections and permissions
3. **Integrate with Raspberry Pi** for production deployment
4. **Connect Flutter app** to the backend API
5. **Deploy and monitor** the complete system

---

*Last Updated: 2024-01-15*  
*Version: 1.0.0*  
*Backend for Bus Safety Gender Detection System*
