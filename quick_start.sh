#!/bin/bash

# Quick Start Script for Gender Detection
echo "🚀 Gender Detection Quick Start"
echo "================================"

# Check if we're in WSL
if grep -q Microsoft /proc/version; then
    echo "✅ Running in WSL"
    echo ""
    echo "Choose your option:"
    echo "1. HTTP (may not work with camera)"
    echo "2. HTTPS (recommended for camera access)"
    echo "3. Pi 5 optimized (for Raspberry Pi 5)"
    echo ""
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo "🌐 Starting HTTP server..."
            python3 start_http.py
            ;;
        2)
            echo "🔒 Starting HTTPS server..."
            python3 start_https.py
            ;;
        3)
            echo "🍓 Starting Pi 5 optimized server..."
            ./start_pi5.sh
            ;;
        *)
            echo "❌ Invalid choice, starting HTTP server..."
            python3 start_http.py
            ;;
    esac
else
    echo "❌ Not running in WSL, please run in WSL environment"
    exit 1
fi

