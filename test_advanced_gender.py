#!/usr/bin/env python3
"""
Test script for Advanced Gender Detection System
"""

import sys
import os
sys.path.append('backend')

def test_advanced_system():
    """Test the advanced gender detection system"""
    print("🧠 Testing Advanced Gender Detection System...")
    print("=" * 50)

    try:
        from backend.live_advanced_gender_detection import LiveAdvancedGenderDetection

        print("✅ Advanced system imported successfully!")

        # Create instance to test initialization
        detector = LiveAdvancedGenderDetection()
        print("✅ Advanced detector initialized!")

        print("\n🎯 System Features:")
        print("   • SCRFD face detection with enhanced validation")
        print("   • Deep Learning gender classification (ResNet50)")
        print("   • Heuristic fallback mode")
        print("   • Real-time performance monitoring")
        print("   • Mode switching (DL vs Heuristic)")
        print("   • Confidence scoring and validation")

        # Test gender classifier separately
        from backend.gender_classifier import create_gender_classifier
        classifier = create_gender_classifier()
        print("✅ Gender classifier loaded!")

        print("\n🚀 Ready to run!")
        print("Run: python backend/live_advanced_gender_detection.py")
        print("\n🎮 Controls:")
        print("   Q: Quit")
        print("   S: Save frame")
        print("   B: Benchmark performance")
        print("   M: Toggle DL/Heuristic mode")
        print("   H: Show help")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_available_models():
    """Show available pre-trained models"""
    print("\n📋 Available Pre-trained Gender Classification Models:")
    print("=" * 60)

    models = {
        'InsightFace Gender': {
            'accuracy': '95%+',
            'description': 'Production-ready gender classifier from InsightFace',
            'url': 'https://github.com/deepinsight/insightface'
        },
        'FairFace': {
            'accuracy': '93%',
            'description': 'Demographic estimation including gender',
            'url': 'https://github.com/dchen236/FairFace'
        },
        'Face Attribute Model': {
            'accuracy': '94%+',
            'description': 'Multi-attribute recognition (gender, age, etc.)',
            'url': 'https://github.com/deepinsight/insightface'
        },
        'CelebA Trained': {
            'accuracy': '96%',
            'description': 'Trained on CelebA dataset (162,770 images)',
            'url': 'https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html'
        },
        'Lightweight MobileNet': {
            'accuracy': '91%',
            'description': 'Fast inference, good for edge devices',
            'url': 'https://pytorch.org/vision/stable/models.html'
        }
    }

    for name, info in models.items():
        print(f"🎯 {name}")
        print(f"   📊 Accuracy: {info['accuracy']}")
        print(f"   📝 {info['description']}")
        print(f"   🔗 {info['url']}")
        print()

def main():
    """Main test function"""
    print("🤖 Advanced Gender Detection System Test")
    print("=" * 50)

    # Test the system
    success = test_advanced_system()

    if success:
        print("✅ All tests passed!")
        show_available_models()

        print("\n💡 Recommendations:")
        print("1. For best accuracy: Use InsightFace or CelebA trained models")
        print("2. For speed: Use MobileNet-based classifiers")
        print("3. For production: Implement model ensemble (multiple models)")

        print("\n🚀 Next Steps:")
        print("1. Download a pre-trained gender model")
        print("2. Run the advanced detection system")
        print("3. Compare DL vs Heuristic accuracy on your webcam")

    else:
        print("❌ Test failed - check dependencies and imports")

if __name__ == "__main__":
    main()
