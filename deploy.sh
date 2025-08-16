#!/bin/bash

echo "🚀 Lambda Cloud Gender Detection - One Command Setup"
echo "=================================================="

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements_cloud.txt

# Check GPU
echo "🔍 Checking GPU..."
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU devices: {len(gpus)}'); [print(f'  {gpu}') for gpu in gpus]"

# Start training
echo "🎯 Starting training..."
python train_modern.py

echo "✅ Deployment complete!"
