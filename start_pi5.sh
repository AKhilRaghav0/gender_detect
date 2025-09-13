#!/bin/bash

# Raspberry Pi 5 Gender Detection Startup Script
# Optimized for 24/7 bus operation

echo "🍓 Starting Gender Detection on Raspberry Pi 5..."

# Set CPU governor to performance for better processing
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase GPU memory split for better performance
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Set process priority
sudo nice -n -10 python3 insightface_web_app.py &

# Get the process ID
PID=$!

# Set up auto-restart on crash
while true; do
    if ! kill -0 $PID 2>/dev/null; then
        echo "Process crashed, restarting..."
        sudo nice -n -10 python3 insightface_web_app.py &
        PID=$!
    fi
    sleep 10
done

