#!/usr/bin/env python3
"""
Master Training Script - All Training Modes
Choose your training strategy and maximize GH200 performance
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Print the training mode selection banner"""
    print("=" * 80)
    print("🚀 GH200 GENDER DETECTION - MASTER TRAINING SCRIPT")
    print("=" * 80)
    print("🔥 Choose your training strategy to maximize GH200 performance!")
    print("💪 Your GH200 has 480GB VRAM - Let's make it sweat!")
    print("=" * 80)

def print_training_modes():
    """Print available training modes"""
    print("\n📚 AVAILABLE TRAINING MODES:")
    print("=" * 50)
    
    modes = [
        {
            'id': '1',
            'name': 'Standard Training',
            'script': 'train_modern_fixed.py',
            'description': 'Basic ResNet50 training (fastest, good baseline)',
            'time': '15-30 minutes',
            'best_for': 'Quick results, baseline performance'
        },
        {
            'id': '2',
            'name': 'K-Fold Cross-Validation',
            'script': 'train_kfold_robust.py',
            'description': '5-fold CV with multiple architectures (maximum robustness)',
            'time': '2-4 hours',
            'best_for': 'Production deployment, maximum reliability'
        },
        {
            'id': '3',
            'name': 'Advanced Augmentation',
            'script': 'train_advanced_augmentation.py',
            'description': 'Intensive data augmentation (maximum data diversity)',
            'time': '1-2 hours',
            'best_for': 'Limited data, maximum generalization'
        },
        {
            'id': '4',
            'name': 'Multi-Model Ensemble',
            'script': 'train_ensemble_models.py',
            'description': 'Train 6 different architectures (maximum performance)',
            'time': '4-6 hours',
            'best_for': 'Best accuracy, research, competition'
        },
        {
            'id': '5',
            'name': 'Run All Modes (Sequential)',
            'script': 'ALL',
            'description': 'Run all training modes one after another',
            'time': '8-12 hours',
            'best_for': 'Complete evaluation, maximum learning'
        }
    ]
    
    for mode in modes:
        print(f"\n{mode['id']}. {mode['name']}")
        print(f"   📝 {mode['description']}")
        print(f"   ⏱️  Estimated time: {mode['time']}")
        print(f"   🎯 Best for: {mode['best_for']}")
    
    print("\n" + "=" * 50)

def check_requirements():
    """Check if all required scripts exist"""
    required_scripts = [
        'train_modern_fixed.py',
        'train_kfold_robust.py',
        'train_advanced_augmentation.py',
        'train_ensemble_models.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print("❌ Missing required training scripts:")
        for script in missing_scripts:
            print(f"   - {script}")
        print("\nPlease ensure all training scripts are in the current directory.")
        return False
    
    print("✅ All required training scripts found!")
    return True

def run_training_script(script_name, mode_name):
    """Run a specific training script"""
    print(f"\n🚀 Starting {mode_name}...")
    print(f"📜 Script: {script_name}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ {mode_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {mode_name} failed with error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script {script_name} not found!")
        return False

def run_all_modes_sequential():
    """Run all training modes sequentially"""
    print("\n🔥 RUNNING ALL TRAINING MODES SEQUENTIALLY!")
    print("🎯 This will take 8-12 hours and create multiple models")
    print("💾 Make sure you have enough disk space!")
    
    confirm = input("\n❓ Are you sure you want to run ALL modes? (yes/no): ").lower()
    if confirm not in ['yes', 'y']:
        print("❌ Cancelled. Choose a single mode instead.")
        return
    
    modes = [
        ('train_modern_fixed.py', 'Standard Training'),
        ('train_kfold_robust.py', 'K-Fold Cross-Validation'),
        ('train_advanced_augmentation.py', 'Advanced Augmentation'),
        ('train_ensemble_models.py', 'Multi-Model Ensemble')
    ]
    
    successful_modes = []
    failed_modes = []
    
    for script, mode_name in modes:
        print(f"\n{'='*60}")
        print(f"🎯 NEXT: {mode_name}")
        print(f"{'='*60}")
        
        if run_training_script(script, mode_name):
            successful_modes.append(mode_name)
        else:
            failed_modes.append(mode_name)
        
        # Wait between modes
        if script != modes[-1][0]:  # Not the last script
            print(f"\n⏳ Waiting 30 seconds before next mode...")
            time.sleep(30)
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 ALL MODES COMPLETED!")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_modes)}")
    for mode in successful_modes:
        print(f"   - {mode}")
    
    if failed_modes:
        print(f"❌ Failed: {len(failed_modes)}")
        for mode in failed_modes:
            print(f"   - {mode}")
    
    print(f"\n🎉 Total successful modes: {len(successful_modes)}/{len(modes)}")

def main():
    """Main function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Cannot proceed without required scripts.")
        return
    
    while True:
        print_training_modes()
        
        try:
            choice = input("\n🎯 Choose training mode (1-5) or 'q' to quit: ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Goodbye! Happy training!")
                break
            
            if choice == '1':
                run_training_script('train_modern_fixed.py', 'Standard Training')
                break
            elif choice == '2':
                run_training_script('train_kfold_robust.py', 'K-Fold Cross-Validation')
                break
            elif choice == '3':
                run_training_script('train_advanced_augmentation.py', 'Advanced Augmentation')
                break
            elif choice == '4':
                run_training_script('train_ensemble_models.py', 'Multi-Model Ensemble')
                break
            elif choice == '5':
                run_all_modes_sequential()
                break
            else:
                print(f"\n❌ Invalid choice: {choice}")
                print("Please choose 1, 2, 3, 4, 5, or 'q' to quit.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue
    
    print(f"\n📊 Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎉 Check the generated models and results!")

if __name__ == "__main__":
    main()
