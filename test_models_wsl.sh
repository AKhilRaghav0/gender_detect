#!/bin/bash
# Test Models in WSL
# This script tests all detection models including InsightFace

echo "🧪 Testing Models in WSL Environment"
echo "======================================"

# Navigate to project directory
cd /mnt/c/Users/Akhil/Documents/GitHub/new/gender_detect

# Activate Python environment
echo "🐍 Activating Python environment..."
source gender_detection_env/bin/activate

# Install InsightFace in WSL (should work without C++ build tools)
echo "📦 Installing InsightFace in WSL..."
pip install insightface

# Test InsightFace availability
echo "🔍 Testing InsightFace availability..."
python -c "
try:
    import insightface
    from insightface.app import FaceAnalysis
    print('✅ InsightFace imported successfully')
    
    # Test initialization
    face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print('✅ InsightFace initialized successfully')
    
    # Cleanup
    del face_app
    print('✅ InsightFace test completed')
    
except Exception as e:
    print(f'❌ InsightFace test failed: {e}')
    exit(1)
"

# Test unified detector with InsightFace
echo "🔍 Testing Unified Detector with InsightFace..."
python BusSaheli/backend/test_insightface_integration.py

# Test memory management
echo "🔍 Testing Memory Management..."
python BusSaheli/backend/working_memory_test.py

# Test API endpoints (if server is running)
echo "🔍 Testing API Endpoints..."
echo "Note: Start API server with: python BusSaheli/backend/api_endpoints.py"
echo "Then run: python BusSaheli/backend/test_api_endpoints.py"

echo "🎉 WSL Testing Complete!"
echo "All models should now work including InsightFace!"
