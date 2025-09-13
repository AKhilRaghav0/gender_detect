# 📱 Bus Saheli Flutter App

## 🎯 App Overview
Mobile application for women's safety in public transport, providing real-time bus safety information and passenger counts.

## 🏗️ App Architecture

### State Management
- **Provider**: For state management
- **Riverpod**: For dependency injection
- **Hive**: For local data storage

### Key Features
1. **Real-time Bus Tracking**: Live passenger counts and safety scores
2. **Route Selection**: Browse available bus routes
3. **Safety Alerts**: Push notifications for safety concerns
4. **Offline Support**: Cached data for offline viewing
5. **User Preferences**: Customizable safety thresholds

## 📱 App Screens

### 1. Splash Screen
- App logo and loading animation
- Check for internet connectivity
- Initialize app state

### 2. Home Screen
```dart
class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Bus Saheli')),
      body: Column(
        children: [
          SafetyOverviewCard(),
          ActiveRoutesList(),
          QuickActions(),
        ],
      ),
    );
  }
}
```

### 3. Route Details Screen
```dart
class RouteDetailsScreen extends StatefulWidget {
  final String routeId;
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Route $routeId')),
      body: Column(
        children: [
          LiveSafetyCard(),
          BusList(),
          SafetyChart(),
          Recommendations(),
        ],
      ),
    );
  }
}
```

### 4. Settings Screen
```dart
class SettingsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: ListView(
        children: [
          NotificationSettings(),
          SafetyThresholds(),
          AboutSection(),
        ],
      ),
    );
  }
}
```

## 🔧 Core Components

### Data Models
```dart
class BusData {
  final String busId;
  final String routeNumber;
  final int passengerCount;
  final int femaleCount;
  final int maleCount;
  final double safetyScore;
  final SafetyLevel safetyLevel;
  final DateTime lastUpdated;
  final bool isActive;
}

enum SafetyLevel { SAFE, MODERATE, UNSAFE }

class RouteData {
  final String routeId;
  final String routeName;
  final List<String> stops;
  final List<BusData> buses;
  final double averageSafetyScore;
}
```

### API Service
```dart
class BusSafetyApiService {
  static const String baseUrl = 'https://api.bussaheli.com';
  
  Future<List<RouteData>> getActiveRoutes() async {
    final response = await http.get(Uri.parse('$baseUrl/api/v1/routes'));
    return RouteData.fromJsonList(response.body);
  }
  
  Future<BusData> getBusData(String busId) async {
    final response = await http.get(Uri.parse('$baseUrl/api/v1/buses/$busId'));
    return BusData.fromJson(response.body);
  }
  
  Stream<BusData> subscribeToBusUpdates(String busId) {
    return WebSocketChannel.connect(
      Uri.parse('wss://api.bussaheli.com/ws/bus-safety')
    ).stream.map((data) => BusData.fromJson(jsonDecode(data)));
  }
}
```

### State Management
```dart
class BusSafetyProvider extends ChangeNotifier {
  final BusSafetyApiService _apiService = BusSafetyApiService();
  
  List<RouteData> _routes = [];
  Map<String, BusData> _busData = {};
  bool _isLoading = false;
  
  List<RouteData> get routes => _routes;
  Map<String, BusData> get busData => _busData;
  bool get isLoading => _isLoading;
  
  Future<void> loadActiveRoutes() async {
    _isLoading = true;
    notifyListeners();
    
    try {
      _routes = await _apiService.getActiveRoutes();
    } catch (e) {
      // Handle error
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }
  
  void subscribeToBusUpdates(String busId) {
    _apiService.subscribeToBusUpdates(busId).listen((data) {
      _busData[busId] = data;
      notifyListeners();
    });
  }
}
```

## 🎨 UI Components

### Safety Indicator Widget
```dart
class SafetyIndicator extends StatelessWidget {
  final SafetyLevel safetyLevel;
  final double safetyScore;
  
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: _getSafetyColor(safetyLevel),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(_getSafetyIcon(safetyLevel)),
          SizedBox(width: 8),
          Text('Safety: ${(safetyScore * 10).toInt()}/10'),
        ],
      ),
    );
  }
  
  Color _getSafetyColor(SafetyLevel level) {
    switch (level) {
      case SafetyLevel.SAFE: return Colors.green;
      case SafetyLevel.MODERATE: return Colors.orange;
      case SafetyLevel.UNSAFE: return Colors.red;
    }
  }
}
```

### Live Bus Card
```dart
class LiveBusCard extends StatelessWidget {
  final BusData busData;
  
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Bus ${busData.busId}', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Passengers: ${busData.passengerCount}'),
                Text('Female: ${busData.femaleCount}'),
              ],
            ),
            SizedBox(height: 8),
            SafetyIndicator(
              safetyLevel: busData.safetyLevel,
              safetyScore: busData.safetyScore,
            ),
          ],
        ),
      ),
    );
  }
}
```

## 🚀 Getting Started

### Prerequisites
- Flutter SDK 3.0+
- Dart 3.0+
- Android Studio / VS Code
- Android device or emulator

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/bus-saheli.git
cd bus-saheli/flutter_app

# Install dependencies
flutter pub get

# Run the app
flutter run
```

### Build for Production
```bash
# Android APK
flutter build apk --release

# Android App Bundle
flutter build appbundle --release

# iOS (requires macOS)
flutter build ios --release
```

## 📊 App Features

### Real-time Updates
- WebSocket connection for live data
- Automatic reconnection on network issues
- Background data refresh

### Offline Support
- Cached route and bus data
- Offline safety recommendations
- Sync when connection restored

### Push Notifications
- Safety alerts for unsafe conditions
- Route updates and changes
- Emergency notifications

### Accessibility
- Screen reader support
- High contrast mode
- Large text options
- Voice navigation

## 🔧 Configuration

### API Configuration
```dart
class ApiConfig {
  static const String baseUrl = 'https://api.bussaheli.com';
  static const String wsUrl = 'wss://api.bussaheli.com/ws';
  static const Duration timeout = Duration(seconds: 30);
}
```

### App Settings
```dart
class AppSettings {
  static const double defaultSafetyThreshold = 0.4;
  static const Duration refreshInterval = Duration(seconds: 30);
  static const bool enableNotifications = true;
  static const bool enableLocationTracking = false;
}
```

## 📱 Platform Support

- **Android**: 5.0+ (API 21+)
- **iOS**: 11.0+
- **Web**: Chrome, Firefox, Safari
- **Desktop**: Windows, macOS, Linux

## 🧪 Testing

### Unit Tests
```bash
flutter test
```

### Integration Tests
```bash
flutter test integration_test/
```

### Widget Tests
```bash
flutter test test/widget_test.dart
```

## 📈 Performance

- **App Size**: < 50MB
- **Startup Time**: < 3 seconds
- **Memory Usage**: < 100MB
- **Battery Impact**: Minimal with optimized refresh rates
