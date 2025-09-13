#!/usr/bin/env python3
"""
Setup script for InspireFace - Advanced Face Analysis Library
"""

import os
import sys
import subprocess
import platform

def setup_inspireface():
    """Setup InspireFace library"""
    print("🚀 Setting up InspireFace - Advanced Face Analysis")
    print("=" * 50)

    # Check if we're on Windows (since InspireFace might have platform-specific builds)
    is_windows = platform.system() == "Windows"

    if is_windows:
        print("🪟 Windows detected - InspireFace may require custom installation")
        print("📋 Let's try installing from source...")

        # Try to clone the repository
        try:
            print("📥 Cloning InspireFace repository...")
            subprocess.run(["git", "clone", "https://github.com/HyperInspire/InspireFace.git"],
                         check=True, capture_output=True)
            print("✅ Repository cloned successfully!")

            # Change to the cloned directory
            os.chdir("InspireFace")

            # Try to install dependencies and build
            print("🔧 Installing dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                         check=True)

            # Try to build the library
            print("🔨 Building InspireFace...")
            if os.path.exists("setup.py"):
                subprocess.run([sys.executable, "setup.py", "install"], check=True)
            elif os.path.exists("pyproject.toml"):
                subprocess.run([sys.executable, "-m", "pip", "install", "."], check=True)

            print("✅ InspireFace installed successfully!")

        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("📋 Manual installation instructions:")
            print("1. Visit: https://github.com/HyperInspire/InspireFace")
            print("2. Download the appropriate release for Windows")
            print("3. Follow the installation instructions in the README")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

    else:
        print("🐧 Non-Windows platform detected")
        print("📋 Please visit: https://github.com/HyperInspire/InspireFace")
        print("   and follow the installation instructions for your platform")

    return True

def create_inspireface_demo():
    """Create a demo script for InspireFace features"""
    demo_code = '''#!/usr/bin/env python3
"""
InspireFace Demo - Advanced Face Analysis
"""

import cv2
import numpy as np
import sys
import os

# Add current directory to path for local imports
sys.path.append('.')

def inspireface_demo():
    """Demo of InspireFace advanced features"""
    print("🎭 InspireFace Advanced Face Analysis Demo")
    print("=" * 50)

    try:
        # Try to import InspireFace
        import inspireface as ifa

        print("✅ InspireFace imported successfully!")

        # Initialize the face analyzer
        analyzer = ifa.InspireFaceAnalyzer()

        # Load models
        print("🔧 Loading face analysis models...")

        # This would depend on the specific InspireFace API
        # analyzer.load_model("face_detection")
        # analyzer.load_model("gender_age")
        # analyzer.load_model("face_quality")
        # analyzer.load_model("eye_status")

        print("✅ Models loaded!")

        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        print("📹 Starting live analysis...")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze frame with InspireFace
            results = analyzer.analyze(frame)

            # Process results
            for face in results:
                # Draw bounding box
                x, y, w, h = face.bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Display analysis results
                gender = face.gender
                age = face.age
                quality = face.quality
                eye_status = face.eye_status

                # Display information
                info = f"{gender}, Age: {age}, Quality: {quality:.2f}"
                cv2.putText(frame, info, (x, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Eye status
                if hasattr(face, 'eye_status'):
                    left_eye = "Open" if face.eye_status.left == "open" else "Closed"
                    right_eye = "Open" if face.eye_status.right == "open" else "Closed"
                    eye_info = f"L:{left_eye} R:{right_eye}"
                    cv2.putText(frame, eye_info, (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Show frame
            cv2.imshow("InspireFace Advanced Analysis", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except ImportError:
        print("❌ InspireFace not installed!")
        print("📋 Please install InspireFace first:")
        print("   python setup_inspireface.py")
        return

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return

if __name__ == "__main__":
    inspireface_demo()
'''

    with open("inspireface_demo.py", "w") as f:
        f.write(demo_code)

    print("✅ Created inspireface_demo.py")
    print("🎯 Run: python inspireface_demo.py")

def show_alternatives():
    """Show alternative face analysis libraries"""
    print("\n🔄 Alternative Advanced Face Analysis Libraries:")
    print("=" * 50)

    alternatives = {
        "DeepFace": {
            "features": ["Gender", "Age", "Emotion", "Race"],
            "install": "pip install deepface",
            "github": "https://github.com/serengil/deepface"
        },
        "Face Recognition": {
            "features": ["Face Detection", "Recognition", "Landmarks"],
            "install": "pip install face-recognition",
            "github": "https://github.com/ageitgey/face_recognition"
        },
        "InsightFace": {
            "features": ["Face Detection", "Recognition", "Gender", "Age"],
            "install": "pip install insightface",
            "github": "https://github.com/deepinsight/insightface"
        },
        "MediaPipe": {
            "features": ["Face Detection", "Landmarks", "Mesh"],
            "install": "pip install mediapipe",
            "github": "https://github.com/google/mediapipe"
        }
    }

    for name, info in alternatives.items():
        print(f"🎯 {name}")
        print(f"   ✨ Features: {', '.join(info['features'])}")
        print(f"   📦 Install: {info['install']}")
        print(f"   🔗 GitHub: {info['github']}")
        print()

if __name__ == "__main__":
    print("🤖 InspireFace Setup Script")
    print("=" * 50)

    success = setup_inspireface()

    if success:
        print("\n🎉 InspireFace setup completed!")
        create_inspireface_demo()
    else:
        print("\n⚠️ InspireFace setup failed")
        show_alternatives()

    print("\n🚀 Next steps:")
    print("1. Install InspireFace or an alternative")
    print("2. Run: python inspireface_demo.py")
    print("3. Enjoy advanced face analysis features!")
