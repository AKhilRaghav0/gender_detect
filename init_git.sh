#!/bin/bash

# Initialize new git repository for Gender Detection Project

echo "🚀 Initializing new Git repository..."

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
gender_detection_env/
venv/
ENV/

# Model files (too large for git)
*.model
*.h5
*.pkl

# Training plots
*.png
*.jpg
*.jpeg

# Logs
*.log
training.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
EOF

# Add all files
git add .

# Initial commit
git commit -m "🚀 Initial commit: Gender Detection Project with VPS support

- CNN-based gender detection using TensorFlow/Keras
- OpenCV for computer vision and face detection
- VPS-optimized training and detection scripts
- Comprehensive installation and deployment guides
- Dataset: gender_dataset_face/ with man/woman categories
- Optimized for Ubuntu/Debian VPS deployment"

echo "✅ Git repository initialized successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Add your remote origin:"
echo "   git remote add origin <your-repo-url>"
echo ""
echo "2. Push to your repository:"
echo "   git push -u origin main"
echo ""
echo "3. Upload to VPS and run:"
echo "   bash install_vps.sh"
echo ""
echo "🎉 Your project is ready for VPS deployment!"
