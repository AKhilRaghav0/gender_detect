#!/bin/bash

# Gender Detection Project - VPS Installation Script
# This script installs all dependencies and sets up the environment on Ubuntu/Debian VPS

echo "🚀 Starting Gender Detection Project VPS Setup..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv git curl wget unzip

# Install OpenCV system dependencies
echo "📷 Installing OpenCV system dependencies..."
sudo apt install -y libopencv-dev python3-opencv libgl1-mesa-glx libglib2.0-0

# Install additional libraries
echo "📚 Installing additional libraries..."
sudo apt install -y libsm6 libxext6 libxrender-dev libgomp1

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv gender_detection_env
source gender_detection_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python packages
echo "📦 Installing Python packages..."
pip install tensorflow==2.15.0
pip install opencv-python==4.7.0.72
pip install opencv-contrib-python==4.7.0.72
pip install cvlib==0.2.7
pip install scikit-learn==1.2.2
pip install matplotlib==3.6.3
pip install numpy>=2.3.2
pip install Pillow>=11.3.0

# Create training script optimized for VPS
echo "📝 Creating optimized training script..."
cat > train_vps.py << 'EOF'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for VPS
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import time

# VPS optimized parameters
epochs = 50  # Reduced for faster training
lr = 1e-3
batch_size = 32  # Reduced for VPS memory
img_dims = (96,96,3)

print(f"🚀 Starting training with {epochs} epochs, batch size {batch_size}")
print(f"📁 Loading images from gender_dataset_face/")

data = []
labels = []

# Load image files from the dataset
image_files = [f for f in glob.glob('gender_dataset_face/**/*', recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

print(f"📊 Found {len(image_files)} images")

# Converting images to arrays and labelling the categories
for i, img in enumerate(image_files):
    if i % 100 == 0:
        print(f"🔄 Processing image {i}/{len(image_files)}")
    
    image = cv2.imread(img)
    if image is None:
        continue
        
    image = cv2.resize(image, (img_dims[0],img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]
    if label == "woman":
        label = 1
    else:
        label = 0
        
    labels.append([label])

print(f"✅ Processed {len(data)} images successfully")

# Pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

print(f"📊 Training set: {len(trainX)} images")
print(f"📊 Validation set: {len(testX)} images")

# Augmenting dataset 
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Define model
def buildModel(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build model
print("🏗️ Building model...")
model = buildModel(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Compile the model
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("📊 Model summary:")
model.summary()

# Train the model
print("🚀 Starting training...")
start_time = time.time()

H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
               validation_data=(testX,testY),
               steps_per_epoch=len(trainX) // batch_size,
               epochs=epochs, verbose=1)

training_time = time.time() - start_time
print(f"⏱️ Training completed in {training_time/60:.2f} minutes")

# Save the model to disk
print("💾 Saving model...")
model.save('gender_detection_vps.model')
print("✅ Model saved as 'gender_detection_vps.model'")

# Plot training/validation loss/accuracy
print("📈 Creating training plots...")
plt.style.use("ggplot")
plt.figure(figsize=(12, 8))

N = epochs
plt.subplot(2, 1, 1)
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig('training_results_vps.png', dpi=300, bbox_inches='tight')
print("✅ Training plots saved as 'training_results_vps.png'")

# Print final results
final_train_acc = H.history["accuracy"][-1]
final_val_acc = H.history["val_accuracy"][-1]
final_train_loss = H.history["loss"][-1]
final_val_loss = H.history["val_loss"][-1]

print("\n🎉 Training Results Summary:")
print(f"📊 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📊 Final Validation Accuracy: {final_val_acc:.4f}")
print(f"📊 Final Training Loss: {final_train_loss:.4f}")
print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
print(f"⏱️ Total Training Time: {training_time/60:.2f} minutes")

print("\n✅ Training completed successfully!")
EOF

# Create VPS webcam detection script
echo "📹 Creating VPS webcam detection script..."
cat > detect_gender_vps.py << 'EOF'
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import time

print("🚀 Loading gender detection model...")
try:
    model = load_model('gender_detection_vps.model')
    print("✅ Model loaded successfully!")
except:
    print("❌ Could not load 'gender_detection_vps.model'")
    print("💡 Please train the model first using: python3 train_vps.py")
    exit(1)

print("📹 Opening webcam...")
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("❌ Could not open webcam")
    exit(1)

print("✅ Webcam opened successfully!")
print("💡 Press 'Q' to quit")
    
classes = ['man','woman']
frame_count = 0
start_time = time.time()

# Loop through frames
while webcam.isOpened():
    # Read frame from webcam 
    status, frame = webcam.read()
    
    if not status:
        print("❌ Could not read frame")
        break

    # Apply face detection
    face, confidence = cv.detect_face(frame)

    # Loop through detected faces
    for idx, f in enumerate(face):
        # Get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop, verbose=0)[0]

        # Get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Gender Detection - VPS", frame)
    
    frame_count += 1
    
    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time if total_time > 0 else 0

print(f"\n📊 Performance Summary:")
print(f"🎬 Total Frames Processed: {frame_count}")
print(f"⏱️ Total Time: {total_time:.2f} seconds")
print(f"🚀 Average FPS: {fps:.2f}")

# Release resources
webcam.release()
cv2.destroyAllWindows()
print("✅ Webcam detection stopped")
EOF

# Create a simple test script
echo "🧪 Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3

print("🧪 Testing VPS Setup...")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow import failed: {e}")
    exit(1)

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__} imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    exit(1)

try:
    import cvlib
    print("✅ cvlib imported successfully")
except ImportError as e:
    print(f"❌ cvlib import failed: {e}")
    exit(1)

try:
    import sklearn
    print(f"✅ scikit-learn {sklearn.__version__} imported successfully")
except ImportError as e:
    print(f"❌ scikit-learn import failed: {e}")
    exit(1)

try:
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__} imported successfully")
except ImportError as e:
    print(f"❌ matplotlib import failed: {e}")
    exit(1)

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")
    exit(1)

print("\n🎉 All packages imported successfully!")
print("✅ VPS setup is ready for training!")

# Test GPU availability
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   - {gpu.name}")
    else:
        print("💻 Using CPU for training")
except:
    print("💻 Using CPU for training")
EOF

# Make scripts executable
chmod +x install_vps.sh
chmod +x train_vps.py
chmod +x detect_gender_vps.py
chmod +x test_setup.py

echo ""
echo "🎉 VPS Setup Complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload this project to your VPS"
echo "2. Run: bash install_vps.sh"
echo "3. Test setup: python3 test_setup.py"
echo "4. Start training: python3 train_vps.py"
echo "5. Run detection: python3 detect_gender_vps.py"
echo ""
echo "💡 The VPS scripts are optimized for:"
echo "   - Reduced memory usage"
echo "   - Non-interactive matplotlib backend"
echo "   - Progress tracking during training"
echo "   - Performance monitoring"
