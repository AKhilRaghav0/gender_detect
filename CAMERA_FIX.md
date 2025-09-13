# 📷 Camera Access Fix Guide

## **Problem**: `Cannot read properties of undefined (reading 'getUserMedia')`

This error occurs because modern browsers require **HTTPS** for camera access.

## **Solutions** (in order of preference):

### **1. 🔒 Use HTTPS (Recommended)**
```bash
# Start HTTPS server
python3 start_https.py
```
- **Access**: `https://172.22.196.192:5000`
- **Mobile**: `https://[YOUR_IP]:5000`
- **Note**: Accept security warning for self-signed certificate

### **2. 🌐 Use HTTP (Fallback)**
```bash
# Start HTTP server
python3 start_http.py
```
- **Access**: `http://172.22.196.192:5000`
- **Note**: Camera may not work in some browsers

### **3. 🍓 Pi 5 Optimized**
```bash
# Start Pi 5 optimized server
./start_pi5.sh
```

### **4. 🚀 Quick Start**
```bash
# Interactive menu
./quick_start.sh
```

## **Browser Compatibility**:

| Browser | HTTP | HTTPS | Notes |
|---------|------|-------|-------|
| Chrome | ❌ | ✅ | Best support |
| Firefox | ❌ | ✅ | Good support |
| Edge | ❌ | ✅ | Good support |
| Safari | ❌ | ✅ | iOS/macOS only |
| Mobile | ❌ | ✅ | Use HTTPS |

## **Troubleshooting**:

### **Error: "Camera Permission Denied"**
- Click the camera icon in browser address bar
- Select "Allow" for camera access
- Refresh the page

### **Error: "No Camera Found"**
- Check if camera is connected
- Try different camera in browser settings
- Restart browser

### **Error: "Camera Not Supported"**
- Use HTTPS instead of HTTP
- Try different browser
- Check browser version

### **Error: "getUserMedia not supported"**
- Update browser to latest version
- Use Chrome, Firefox, or Edge
- Enable JavaScript

## **For Production**:
- Use a real SSL certificate (not self-signed)
- Set up proper domain name
- Configure reverse proxy (nginx/apache)

## **Mobile Access**:
1. Find your computer's IP address
2. Use `https://[IP]:5000` on phone
3. Accept security warning
4. Allow camera access

