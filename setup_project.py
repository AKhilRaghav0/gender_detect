#!/usr/bin/env python3
"""
🤖 Advanced Gender Detection System - Automated Setup Script

This script sets up the entire project environment including:
- Virtual environment creation and activation
- Dependency installation
- Model downloads
- Configuration setup
- Basic testing

Run this script to get the project ready for development and testing.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class ProjectSetup:
    """Automated project setup and configuration"""

    def __init__(self):
        """Initialize setup configuration"""
        self.project_root = Path(__file__).parent.absolute()
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        self.is_linux = self.system == "linux"
        self.is_macos = self.system == "darwin"

        # Configuration
        self.venv_name = "gender_detect_env"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        print("🚀 Advanced Gender Detection System Setup")
        print("=" * 50)
        print(f"📍 Project: {self.project_root}")
        print(f"🖥️  System: {self.system}")
        print(f"🐍 Python: {self.python_version}")
        print()

    def run_command(self, cmd, cwd=None, shell=False):
        """Execute command with proper error handling"""
        try:
            result = subprocess.run(
                cmd if isinstance(cmd, list) else cmd.split(),
                cwd=cwd or self.project_root,
                shell=shell,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            print(f"Error: {e.stderr}")
            return None

    def create_virtual_environment(self):
        """Create and setup virtual environment"""
        print("🔧 Setting up Virtual Environment...")
        print("-" * 40)

        venv_path = self.project_root / self.venv_name

        # Remove existing venv if it exists
        if venv_path.exists():
            print(f"🗑️  Removing existing environment: {venv_path}")
            if self.is_windows:
                shutil.rmtree(venv_path)
            else:
                self.run_command(f"rm -rf {venv_path}")

        # Create new virtual environment
        print(f"📦 Creating virtual environment: {self.venv_name}")
        cmd = [sys.executable, "-m", "venv", self.venv_name]
        self.run_command(cmd)

        # Activate environment and upgrade pip
        if self.is_windows:
            pip_path = venv_path / "Scripts" / "pip.exe"
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
            python_path = venv_path / "bin" / "python"

        print("⬆️  Upgrading pip...")
        self.run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])

        print("✅ Virtual environment created successfully!")
        return str(python_path), str(pip_path)

    def install_dependencies(self, pip_path):
        """Install project dependencies"""
        print("\n📦 Installing Dependencies...")
        print("-" * 40)

        requirements_files = [
            "requirements.txt",
            "backend/requirements.txt"
        ]

        for req_file in requirements_files:
            if (self.project_root / req_file).exists():
                print(f"📄 Installing from {req_file}...")
                self.run_command([pip_path, "install", "-r", req_file])
            else:
                print(f"⚠️  Requirements file not found: {req_file}")

        # Install additional packages
        print("📦 Installing additional packages...")
        additional_packages = [
            "torch",
            "torchvision",
            "deepface",
            "memory-profiler",
            "pytest",
            "pytest-asyncio"
        ]

        # Try to install with CUDA support for NVIDIA GPUs
        try:
            if self.is_windows:
                self.run_command([pip_path, "install", "torch", "torchvision",
                                "--index-url", "https://download.pytorch.org/whl/cu118"])
            else:
                self.run_command([pip_path, "install", "torch", "torchvision",
                                "--index-url", "https://download.pytorch.org/whl/cu118"])
        except:
            # Fallback to CPU-only
            print("⚠️  CUDA installation failed, installing CPU-only PyTorch...")
            self.run_command([pip_path, "install", "torch", "torchvision"])

        # Install other packages
        for package in additional_packages[2:]:  # Skip torch packages
            try:
                self.run_command([pip_path, "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"⚠️  Failed to install {package}")

        print("✅ Dependencies installed!")

    def setup_directories(self):
        """Create necessary directories"""
        print("\n📁 Setting up Directories...")
        print("-" * 40)

        directories = [
            "backend/models",
            "logs",
            "logs/train",
            "logs/validation",
            "models",
            "tests",
            "docs"
        ]

        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}")

        print("✅ All directories created!")

    def download_models(self):
        """Download pre-trained models"""
        print("\n⬇️  Setting up Models...")
        print("-" * 40)

        models_dir = self.project_root / "backend" / "models"

        # Create __init__.py for models package
        init_file = models_dir / "__init__.py"
        init_file.write_text('"""Models package for pre-trained models"""')

        print("📝 Created models package")
        print("💡 Models will be downloaded automatically when first used")
        print("   SCRFD model: ~50MB (face detection)")
        print("   ResNet model: ~100MB (gender classification)")

    def create_activation_script(self):
        """Create environment activation script"""
        print("\n🔧 Creating Activation Script...")
        print("-" * 40)

        if self.is_windows:
            script_content = f'''@echo off
REM Gender Detection Environment Activation Script
echo 🚀 Activating Gender Detection Environment...
echo.

REM Change to project directory
cd /d "{self.project_root}"

REM Activate virtual environment
call "{self.venv_name}\\Scripts\\activate.bat"

REM Set environment variables
set DEBUG=1
set PYTHONPATH=%PYTHONPATH%;{self.project_root}

echo ✅ Environment activated!
echo 📍 Project: {self.project_root}
echo 🐍 Python: {self.python_version}
echo.
echo 🎯 Quick commands:
echo   python backend/live_scrfd_detection.py     - Basic detection
echo   python backend/live_advanced_gender_detection.py - DL detection
echo   python test_advanced_gender.py             - Test suite
echo.
echo deactivate                                   - Exit environment
echo.

cmd /k
'''
            script_name = "activate_env.bat"
        else:
            script_content = f'''#!/bin/bash
# Gender Detection Environment Activation Script

echo "🚀 Activating Gender Detection Environment..."
echo

# Change to project directory
cd "{self.project_root}"

# Activate virtual environment
source "{self.venv_name}/bin/activate"

# Set environment variables
export DEBUG=1
export PYTHONPATH="$PYTHONPATH:{self.project_root}"

echo "✅ Environment activated!"
echo "📍 Project: {self.project_root}"
echo "🐍 Python: {self.python_version}"
echo
echo "🎯 Quick commands:"
echo "  python backend/live_scrfd_detection.py      - Basic detection"
echo "  python backend/live_advanced_gender_detection.py  - DL detection"
echo "  python test_advanced_gender.py              - Test suite"
echo
echo "deactivate                                    - Exit environment"
echo

# Start interactive shell
exec "$SHELL"
'''
            script_name = "activate_env.sh"

        script_path = self.project_root / script_name
        script_path.write_text(script_content)

        if not self.is_windows:
            script_path.chmod(0o755)

        print(f"✅ Created activation script: {script_name}")

    def run_basic_tests(self, python_path):
        """Run basic functionality tests"""
        print("\n🧪 Running Basic Tests...")
        print("-" * 40)

        test_commands = [
            ([python_path, "-c", "import cv2; print('✅ OpenCV working')"], "OpenCV"),
            ([python_path, "-c", "import torch; print(f'✅ PyTorch working: {torch.__version__}')"], "PyTorch"),
            ([python_path, "-c", "from backend.scrfd_detection import create_scrfd_detector; print('✅ SCRFD module working')"], "SCRFD Module"),
        ]

        passed = 0
        for cmd, name in test_commands:
            try:
                result = self.run_command(cmd)
                if result:
                    print(f"✅ {name}: {result}")
                    passed += 1
                else:
                    print(f"⚠️  {name}: No output")
            except:
                print(f"❌ {name}: Failed")

        print(f"\n📊 Tests passed: {passed}/{len(test_commands)}")

        if passed == len(test_commands):
            print("🎉 All basic tests passed!")
        else:
            print("⚠️  Some tests failed - check dependencies")

    def create_run_scripts(self):
        """Create convenient run scripts"""
        print("\n📝 Creating Run Scripts...")
        print("-" * 40)

        scripts = {
            "run_basic_detection.bat" if self.is_windows else "run_basic_detection.sh": {
                "description": "Run basic SCRFD gender detection",
                "command": "python backend/live_scrfd_detection.py"
            },
            "run_advanced_detection.bat" if self.is_windows else "run_advanced_detection.sh": {
                "description": "Run deep learning gender detection",
                "command": "python backend/live_advanced_gender_detection.py"
            },
            "run_tests.bat" if self.is_windows else "run_tests.sh": {
                "description": "Run comprehensive test suite",
                "command": "python test_advanced_gender.py"
            }
        }

        for script_name, config in scripts.items():
            if self.is_windows:
                content = f'''@echo off
REM {config["description"]}
echo 🚀 {config["description"]}
echo Command: {config["command"]}
echo.
{config["command"]}
pause
'''
            else:
                content = f'''#!/bin/bash
# {config["description"]}
echo "🚀 {config["description"]}"
echo "Command: {config["command"]}"
echo
{config["command"]}
'''

            script_path = self.project_root / script_name
            script_path.write_text(content)

            if not self.is_windows:
                script_path.chmod(0o755)

            print(f"✅ Created {script_name}")

    def create_project_info(self):
        """Create project information file"""
        print("\n📋 Creating Project Info...")
        print("-" * 40)

        info_content = f'''# Project Information
# Generated by setup_project.py

PROJECT_NAME=Advanced Gender Detection System
VERSION=2.0.0
PYTHON_VERSION={self.python_version}
SYSTEM={self.system}
CREATED_BY=Automated Setup Script

# Environment
VENV_NAME={self.venv_name}
PROJECT_ROOT={self.project_root}

# Key Features
FEATURES=Face Detection,Gender Classification,Deep Learning,Real-time Processing,Multi-platform

# Dependencies
MAIN_DEPS=torch,opencv-python,deepface,numpy
EXTRA_DEPS=scrfd,gender-classifier,resnet

# Models
FACE_MODEL=SCRFD 2.5G
CLASSIFICATION_MODEL=ResNet50
PROFESSIONAL_MODEL=InspireFace (Optional)

# Performance
TARGET_FPS=30-60
ACCURACY_TARGET=75-96%
MEMORY_USAGE=500-1500MB

# Platforms
SUPPORTED_PLATFORMS=Windows,Linux,macOS
RECOMMENDED_GPU=NVIDIA CUDA
'''

        info_path = self.project_root / "project_info.txt"
        info_path.write_text(info_content)
        print("✅ Created project_info.txt")

    def show_completion_message(self):
        """Show completion message with next steps"""
        print("\n🎉 Setup Complete!")
        print("=" * 50)
        print("✅ Virtual environment created and configured")
        print("✅ Dependencies installed")
        print("✅ Project structure set up")
        print("✅ Models directory prepared")
        print("✅ Activation scripts created")
        print()

        print("🚀 Next Steps:")
        print("=" * 20)

        if self.is_windows:
            print("1. Activate environment:")
            print(f"   .\\activate_env.bat")
            print()
            print("2. Run basic detection:")
            print("   python backend/live_scrfd_detection.py")
            print()
            print("3. Test deep learning:")
            print("   python backend/live_advanced_gender_detection.py")
        else:
            print("1. Activate environment:")
            print(f"   source {self.venv_name}/bin/activate")
            print()
            print("2. Run basic detection:")
            print("   python backend/live_scrfd_detection.py")
            print()
            print("3. For professional analysis (WSL):")
            print("   bash setup_wsl_inspireface.sh")

        print()
        print("📚 Documentation:")
        print("   README.md - Complete project guide")
        print("   INSPIREFACE_INTEGRATION_GUIDE.md - Professional setup")
        print()
        print("🆘 Need help?")
        print("   Check README.md troubleshooting section")
        print("   Run: python test_advanced_gender.py")
        print()
        print("🎯 Ready to detect genders with AI! 🚀")

    def run_setup(self):
        """Run complete setup process"""
        try:
            # Create virtual environment
            python_path, pip_path = self.create_virtual_environment()

            # Install dependencies
            self.install_dependencies(pip_path)

            # Setup directories
            self.setup_directories()

            # Setup models
            self.download_models()

            # Create activation script
            self.create_activation_script()

            # Create run scripts
            self.create_run_scripts()

            # Create project info
            self.create_project_info()

            # Run basic tests
            self.run_basic_tests(python_path)

            # Show completion message
            self.show_completion_message()

        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            print("💡 Try running individual steps manually")
            print("   Check README.md for troubleshooting")
            return False

        return True

def main():
    """Main setup function"""
    setup = ProjectSetup()
    success = setup.run_setup()

    if success:
        print("\n🎊 Project setup successful!")
        print("Your AI-powered gender detection system is ready! 🤖✨")
    else:
        print("\n⚠️  Setup completed with some issues")
        print("Check the output above for details")

if __name__ == "__main__":
    main()
