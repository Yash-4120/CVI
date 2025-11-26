
"""
CVI620 Assignment 2 - Q1: Cat vs Dog Classification
Inference script for testing on internet images
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import glob

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = 'mobilenet_best.h5'  # Using the best model (100% accuracy)

print("="*60)
print("CAT VS DOG CLASSIFIER - TESTING ON INTERNET IMAGES")
print("="*60)

# Load the trained model
print(f"\nLoading model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nAvailable models in current directory:")
    for f in glob.glob("*.h5"):
        print(f"  - {f}")
    exit(1)

def predict_image(img_path, model):
    """Predict if an image is a cat or dog"""
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret result (0=Cat, 1=Dog based on alphabetical folder order)
    if prediction > 0.5:
        label = "Dog"
        confidence = prediction * 100
    else:
        label = "Cat"
        confidence = (1 - prediction) * 100
    
    return label, confidence, prediction

# Test folder
test_folder = 'internet_images'

# Check if folder exists
if not os.path.exists(test_folder):
    print(f"\n✗ Error: Folder '{test_folder}' not found!")
    print(f"Please create the folder and add some cat/dog images.")
    exit(1)

# Get all image files
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(test_folder, ext)))

if not image_files:
    print(f"\n✗ No images found in '{test_folder}' folder!")
    print("Please add some .jpg or .png images to test.")
    exit(1)

print(f"\n✓ Found {len(image_files)} images to test")
print("="*60)

# Test each image
results = []
for idx, img_path in enumerate(image_files, 1):
    try:
        label, confidence, raw_pred = predict_image(img_path, model)
        results.append((os.path.basename(img_path), label, confidence, raw_pred))
        
        print(f"{idx}. {os.path.basename(img_path):30s} -> {label:4s} ({confidence:5.1f}%)")
        
    except Exception as e:
        print(f"{idx}. {os.path.basename(img_path):30s} -> ERROR: {str(e)}")

# Visualize results
print("\n" + "="*60)
print("GENERATING VISUALIZATION")
print("="*60)

n_images = len(image_files)
cols = min(4, n_images)
rows = (n_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
if n_images == 1:
    axes = np.array([axes])
axes = axes.flatten()

for idx, (img_path, result) in enumerate(zip(image_files, results)):
    try:
        img_name, label, confidence, raw_pred = result
        
        # Load and display image
        img = image.load_img(img_path)
        axes[idx].imshow(img)
        
        # Color: green for high confidence, orange for medium, red for low
        if confidence >= 80:
            color = 'green'
        elif confidence >= 60:
            color = 'orange'
        else:
            color = 'red'
        
        axes[idx].set_title(f'{label}\nConfidence: {confidence:.1f}%', 
                          fontsize=12, fontweight='bold', color=color)
        axes[idx].axis('off')
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'Error loading\n{img_name}', 
                      ha='center', va='center')
        axes[idx].axis('off')

# Hide unused subplots
for idx in range(n_images, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Cat vs Dog Predictions on Internet Images', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('internet_images_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Results saved to 'internet_images_results.png'")

# Show the plot
plt.show()

# Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
cat_count = sum(1 for r in results if r[1] == 'Cat')
dog_count = sum(1 for r in results if r[1] == 'Dog')
avg_confidence = np.mean([r[2] for r in results])

print(f"Total images tested: {len(results)}")
print(f"Predicted as Cat:    {cat_count}")
print(f"Predicted as Dog:    {dog_count}")
print(f"Average confidence:  {avg_confidence:.1f}%")
print("\n" + "="*60)
print("TESTING COMPLETE!")
print("="*60)
print("\nNext: Add 'internet_images_results.png' to your assignment document!")