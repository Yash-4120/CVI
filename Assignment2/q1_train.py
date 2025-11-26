"""
CVI620 Assignment 2 - Q1: Cat vs Dog Classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'Q1/train'
TEST_DIR = 'Q1/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Classes: {train_generator.class_indices}")


# Method 1: Simple CNN
def create_simple_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Method 2: VGG16 Transfer Learning
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Method 3: ResNet50 Transfer Learning
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Method 4: MobileNetV2 Transfer Learning (Best for efficiency)
def create_mobilenet_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(model, model_name):
    """Train a model and save the best weights"""
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n{model_name} Test Accuracy: {test_acc*100:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()
    
    return model, test_acc


# Train all models
results = {}

# 1. Simple CNN
print("\n" + "="*60)
print("METHOD 1: Simple CNN")
print("="*60)
model_simple = create_simple_cnn()
model_simple.summary()
model_simple, acc_simple = train_model(model_simple, 'simple_cnn')
results['Simple CNN'] = acc_simple

# 2. VGG16
print("\n" + "="*60)
print("METHOD 2: VGG16 Transfer Learning")
print("="*60)
model_vgg = create_vgg16_model()
model_vgg, acc_vgg = train_model(model_vgg, 'vgg16')
results['VGG16'] = acc_vgg

# 3. ResNet50
print("\n" + "="*60)
print("METHOD 3: ResNet50 Transfer Learning")
print("="*60)
model_resnet = create_resnet50_model()
model_resnet, acc_resnet = train_model(model_resnet, 'resnet50')
results['ResNet50'] = acc_resnet

# 4. MobileNetV2
print("\n" + "="*60)
print("METHOD 4: MobileNetV2 Transfer Learning")
print("="*60)
model_mobile = create_mobilenet_model()
model_mobile, acc_mobile = train_model(model_mobile, 'mobilenet')
results['MobileNetV2'] = acc_mobile

# Compare results
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)
for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {accuracy*100:.2f}%")

best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with {results[best_model]*100:.2f}% accuracy")

# Plot comparison
plt.figure(figsize=(10, 6))
models = list(results.keys())
accuracies = [results[m]*100 for m in models]
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'orange'])
plt.ylabel('Test Accuracy (%)')
plt.title('Model Comparison - Cat vs Dog Classification')
plt.ylim(0, 100)
for i, v in enumerate(accuracies):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nTraining complete! Models and plots saved.")