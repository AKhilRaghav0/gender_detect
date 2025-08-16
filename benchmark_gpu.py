#!/usr/bin/env python3
"""
GPU Benchmark and Performance Testing for Gender Detection
Tests NVIDIA RTX 3050, GTX 1200 series, and other GPUs
"""

import time
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_gpu_availability():
    """Check GPU availability and details"""
    print("🔍 Checking GPU availability...")
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow GPU support
        print(f"📦 TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            
            gpu_info = []
            for i, gpu in enumerate(gpus):
                try:
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Get GPU details
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                    
                    print(f"  GPU {i}: {gpu_name}")
                    
                    # Determine GPU category
                    if 'RTX 3050' in gpu_name:
                        category = 'RTX_3050'
                        expected_performance = 'Excellent'
                    elif 'GTX 12' in gpu_name:
                        category = 'GTX_1200'
                        expected_performance = 'Good'
                    elif 'RTX' in gpu_name:
                        category = 'RTX_HIGH_END'
                        expected_performance = 'Excellent'
                    elif 'GTX' in gpu_name:
                        category = 'GTX_SERIES'
                        expected_performance = 'Good'
                    else:
                        category = 'OTHER'
                        expected_performance = 'Unknown'
                    
                    gpu_info.append({
                        'index': i,
                        'name': gpu_name,
                        'category': category,
                        'expected_performance': expected_performance
                    })
                    
                    print(f"    Category: {category}")
                    print(f"    Expected Performance: {expected_performance}")
                    
                except Exception as e:
                    print(f"    Error getting details: {e}")
            
            return gpu_info
        else:
            print("❌ No GPUs detected")
            return []
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return []
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return []

def benchmark_basic_operations():
    """Benchmark basic TensorFlow operations"""
    print("\n🏃 Running basic operations benchmark...")
    
    try:
        import tensorflow as tf
        
        # Test data
        batch_sizes = [1, 8, 16, 32, 64]
        image_size = 224
        
        results = {
            'cpu': {},
            'gpu': {}
        }
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Create test data
            test_data = tf.random.normal((batch_size, image_size, image_size, 3))
            
            # CPU test
            with tf.device('/CPU:0'):
                start_time = time.time()
                
                # Simple convolution
                conv_layer = tf.keras.layers.Conv2D(64, 3, activation='relu')
                result_cpu = conv_layer(test_data)
                _ = result_cpu.numpy()  # Force execution
                
                cpu_time = time.time() - start_time
                results['cpu'][batch_size] = cpu_time
                print(f"    CPU: {cpu_time:.4f}s")
            
            # GPU test (if available)
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                with tf.device('/GPU:0'):
                    start_time = time.time()
                    
                    conv_layer_gpu = tf.keras.layers.Conv2D(64, 3, activation='relu')
                    result_gpu = conv_layer_gpu(test_data)
                    _ = result_gpu.numpy()  # Force execution
                    
                    gpu_time = time.time() - start_time
                    results['gpu'][batch_size] = gpu_time
                    print(f"    GPU: {gpu_time:.4f}s")
                    
                    if gpu_time > 0:
                        speedup = cpu_time / gpu_time
                        print(f"    Speedup: {speedup:.2f}x")
        
        return results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {}

def benchmark_resnet_inference():
    """Benchmark ResNet50 inference performance"""
    print("\n🏗️  Running ResNet50 inference benchmark...")
    
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        
        # Load ResNet50
        model = ResNet50(weights='imagenet', include_top=True)
        print("✅ ResNet50 loaded")
        
        # Test configurations
        configs = [
            {'batch_size': 1, 'iterations': 10},
            {'batch_size': 8, 'iterations': 5},
            {'batch_size': 16, 'iterations': 3},
            {'batch_size': 32, 'iterations': 2}
        ]
        
        results = {}
        
        for config in configs:
            batch_size = config['batch_size']
            iterations = config['iterations']
            
            print(f"  Testing batch size {batch_size} ({iterations} iterations)...")
            
            # Create test data
            test_data = tf.random.normal((batch_size, 224, 224, 3))
            
            # Warmup
            _ = model(test_data)
            
            # CPU benchmark
            with tf.device('/CPU:0'):
                start_time = time.time()
                for _ in range(iterations):
                    _ = model(test_data)
                cpu_time = (time.time() - start_time) / iterations
                
                print(f"    CPU: {cpu_time:.4f}s per batch")
            
            # GPU benchmark
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                with tf.device('/GPU:0'):
                    start_time = time.time()
                    for _ in range(iterations):
                        _ = model(test_data)
                    gpu_time = (time.time() - start_time) / iterations
                    
                    print(f"    GPU: {gpu_time:.4f}s per batch")
                    
                    if gpu_time > 0:
                        speedup = cpu_time / gpu_time
                        print(f"    Speedup: {speedup:.2f}x")
                        
                        # Calculate FPS
                        fps = batch_size / gpu_time
                        print(f"    GPU FPS: {fps:.1f}")
                    
                    results[batch_size] = {
                        'cpu_time': cpu_time,
                        'gpu_time': gpu_time,
                        'speedup': cpu_time / gpu_time if gpu_time > 0 else 0,
                        'fps': batch_size / gpu_time if gpu_time > 0 else 0
                    }
            else:
                results[batch_size] = {
                    'cpu_time': cpu_time,
                    'gpu_time': None,
                    'speedup': None,
                    'fps': batch_size / cpu_time
                }
        
        return results
        
    except Exception as e:
        print(f"❌ ResNet benchmark failed: {e}")
        return {}

def benchmark_mixed_precision():
    """Benchmark mixed precision training"""
    print("\n⚡ Testing mixed precision performance...")
    
    try:
        import tensorflow as tf
        
        # Check if mixed precision is supported
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ Mixed precision requires GPU")
            return {}
        
        # Test model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Test data
        batch_size = 32
        test_data = tf.random.normal((batch_size, 224, 224, 3))
        test_labels = tf.random.uniform((batch_size,), maxval=2, dtype=tf.int32)
        
        results = {}
        
        # Standard precision test
        print("  Testing standard (float32) precision...")
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        
        start_time = time.time()
        for _ in range(5):
            with tf.GradientTape() as tape:
                predictions = model(test_data)
                loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        float32_time = time.time() - start_time
        print(f"    Float32 time: {float32_time:.4f}s")
        
        # Mixed precision test
        print("  Testing mixed (float16) precision...")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Recreate model with mixed precision
            model_fp16 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax', dtype='float32')  # Keep output as float32
            ])
            
            optimizer_fp16 = tf.keras.mixed_precision.LossScaleOptimizer(
                tf.keras.optimizers.Adam()
            )
            model_fp16.compile(optimizer=optimizer_fp16, loss='sparse_categorical_crossentropy')
            
            start_time = time.time()
            for _ in range(5):
                with tf.GradientTape() as tape:
                    predictions = model_fp16(test_data)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(test_labels, predictions)
                    scaled_loss = optimizer_fp16.get_scaled_loss(loss)
                
                scaled_gradients = tape.gradient(scaled_loss, model_fp16.trainable_variables)
                gradients = optimizer_fp16.get_unscaled_gradients(scaled_gradients)
                optimizer_fp16.apply_gradients(zip(gradients, model_fp16.trainable_variables))
            
            float16_time = time.time() - start_time
            print(f"    Float16 time: {float16_time:.4f}s")
            
            speedup = float32_time / float16_time if float16_time > 0 else 0
            print(f"    Mixed precision speedup: {speedup:.2f}x")
            
            results = {
                'float32_time': float32_time,
                'float16_time': float16_time,
                'speedup': speedup
            }
            
            # Reset policy
            tf.keras.mixed_precision.set_global_policy('float32')
            
        except Exception as e:
            print(f"    Mixed precision not supported: {e}")
            results = {
                'float32_time': float32_time,
                'float16_time': None,
                'speedup': None
            }
        
        return results
        
    except Exception as e:
        print(f"❌ Mixed precision benchmark failed: {e}")
        return {}

def benchmark_memory_usage():
    """Benchmark GPU memory usage"""
    print("\n💾 Testing GPU memory usage...")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ GPU memory test requires GPU")
            return {}
        
        # Test different batch sizes and their memory usage
        batch_sizes = [1, 8, 16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            try:
                # Create model
                model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dense(2, activation='softmax')
                ])
                
                # Test data
                test_data = tf.random.normal((batch_size, 224, 224, 3))
                
                # Forward pass
                with tf.device('/GPU:0'):
                    start_time = time.time()
                    predictions = model(test_data)
                    inference_time = time.time() - start_time
                
                results[batch_size] = {
                    'inference_time': inference_time,
                    'fps': batch_size / inference_time,
                    'success': True
                }
                
                print(f"    ✅ Success - {inference_time:.4f}s ({batch_size/inference_time:.1f} FPS)")
                
            except tf.errors.ResourceExhaustedError:
                print(f"    ❌ Out of memory")
                results[batch_size] = {
                    'inference_time': None,
                    'fps': None,
                    'success': False,
                    'error': 'Out of memory'
                }
                break
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[batch_size] = {
                    'inference_time': None,
                    'fps': None,
                    'success': False,
                    'error': str(e)
                }
        
        return results
        
    except Exception as e:
        print(f"❌ Memory benchmark failed: {e}")
        return {}

def generate_report(gpu_info, basic_results, resnet_results, mixed_precision_results, memory_results):
    """Generate comprehensive benchmark report"""
    print("\n📊 Generating benchmark report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_info': gpu_info,
        'basic_operations': basic_results,
        'resnet_inference': resnet_results,
        'mixed_precision': mixed_precision_results,
        'memory_usage': memory_results
    }
    
    # Save detailed report
    with open('gpu_benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print("💾 Detailed report saved to gpu_benchmark_report.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 60)
    
    if gpu_info:
        for gpu in gpu_info:
            print(f"🎮 GPU: {gpu['name']}")
            print(f"   Category: {gpu['category']}")
            print(f"   Expected Performance: {gpu['expected_performance']}")
    else:
        print("🖥️  No GPU detected - CPU only")
    
    # ResNet performance summary
    if resnet_results:
        print(f"\n🏗️  ResNet50 Performance:")
        best_fps = 0
        best_batch = 0
        
        for batch_size, results in resnet_results.items():
            fps = results.get('fps', 0)
            if fps > best_fps:
                best_fps = fps
                best_batch = batch_size
        
        if best_fps > 0:
            print(f"   Best FPS: {best_fps:.1f} (batch size {best_batch})")
            
            if best_fps >= 60:
                print("   ✅ Excellent for real-time applications")
            elif best_fps >= 30:
                print("   ✅ Good for real-time applications")
            else:
                print("   ⚠️  May struggle with real-time applications")
    
    # Mixed precision summary
    if mixed_precision_results:
        speedup = mixed_precision_results.get('speedup')
        if speedup:
            print(f"\n⚡ Mixed Precision Speedup: {speedup:.2f}x")
            if speedup >= 1.5:
                print("   ✅ Significant speedup available")
            elif speedup >= 1.2:
                print("   ✅ Moderate speedup available")
            else:
                print("   ℹ️  Minimal speedup")
    
    # Memory recommendations
    if memory_results:
        max_successful_batch = 0
        for batch_size, result in memory_results.items():
            if result.get('success', False):
                max_successful_batch = max(max_successful_batch, batch_size)
        
        if max_successful_batch > 0:
            print(f"\n💾 Maximum Batch Size: {max_successful_batch}")
            
            if max_successful_batch >= 64:
                print("   ✅ High memory capacity - can use large batches")
            elif max_successful_batch >= 32:
                print("   ✅ Good memory capacity - standard batches work well")
            else:
                print("   ⚠️  Limited memory - use smaller batches")
    
    print("\n💡 Recommendations:")
    
    if gpu_info:
        gpu = gpu_info[0]  # Primary GPU
        if gpu['category'] == 'RTX_3050':
            print("   - Use batch size 64 for training")
            print("   - Enable mixed precision for 2x speedup")
            print("   - Real-time webcam detection should work excellently")
        elif gpu['category'] == 'GTX_1200':
            print("   - Use batch size 32 for training")
            print("   - Mixed precision may not be available")
            print("   - Real-time webcam detection should work well")
        elif 'RTX' in gpu['category']:
            print("   - Use batch size 128+ for training")
            print("   - Enable mixed precision for maximum performance")
            print("   - Excellent for all applications")
        else:
            print("   - Start with batch size 32 and adjust based on memory")
            print("   - Test mixed precision if available")
    else:
        print("   - Use batch size 16-32 for CPU training")
        print("   - Consider using fewer epochs")
        print("   - Real-time detection may be slower")
    
    print("=" * 60)

def main():
    """Main benchmark function"""
    print("=" * 60)
    print("🚀 GPU Benchmark for Gender Detection")
    print("Testing NVIDIA RTX 3050, GTX 1200 series, and more")
    print("=" * 60)
    
    # Check GPU availability
    gpu_info = check_gpu_availability()
    
    # Run benchmarks
    basic_results = benchmark_basic_operations()
    resnet_results = benchmark_resnet_inference()
    mixed_precision_results = benchmark_mixed_precision()
    memory_results = benchmark_memory_usage()
    
    # Generate report
    generate_report(gpu_info, basic_results, resnet_results, 
                   mixed_precision_results, memory_results)

if __name__ == "__main__":
    main()
