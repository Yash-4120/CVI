"""
CVI620 Assignment 2 - Q2: MNIST Digit Classification
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("CVI620 - Q2: MNIST DIGIT CLASSIFICATION")
print("="*70)

# Load data
print("\nLoading MNIST data from CSV files...")
train_df = pd.read_csv('Q2/mnist_train.csv')
test_df = pd.read_csv('Q2/mnist_test.csv')

# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

print(f"✓ Training samples: {X_train.shape[0]:,}")
print(f"✓ Test samples: {X_test.shape[0]:,}")
print(f"✓ Image size: 28x28 pixels")
print(f"✓ Classes: {sorted(np.unique(y_train))}")

# Normalize data
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

# Reshape for CNN
X_train_cnn = X_train_norm.reshape(-1, 28, 28, 1)
X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)

# Results storage
results = {}
training_times = {}
models_dict = {}

print("\n" + "="*70)
print("TRAINING MULTIPLE MODELS")
print("="*70)


# ===========================================================================
# METHOD 1: Simple Deep Neural Network
# ===========================================================================
print("\n" + "-"*70)
print("METHOD 1: Simple Deep Neural Network")
print("-"*70)

def create_simple_nn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model

start_time = time.time()
model_nn = create_simple_nn()
model_nn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Simple NN...")
history_nn = model_nn.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

loss, acc = model_nn.evaluate(X_test_cnn, y_test, verbose=0)
training_times['Simple NN'] = time.time() - start_time
results['Simple NN'] = acc
models_dict['Simple NN'] = model_nn
model_nn.save('mnist_simple_nn.keras')
print(f"✓ Accuracy: {acc*100:.2f}% | Time: {training_times['Simple NN']:.1f}s")


# ===========================================================================
# METHOD 2: Convolutional Neural Network (CNN)
# ===========================================================================
print("\n" + "-"*70)
print("METHOD 2: Convolutional Neural Network (CNN)")
print("-"*70)

def create_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

start_time = time.time()
model_cnn = create_cnn()
model_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training CNN...")
history_cnn = model_cnn.fit(
    X_train_cnn, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

loss, acc = model_cnn.evaluate(X_test_cnn, y_test, verbose=0)
training_times['CNN'] = time.time() - start_time
results['CNN'] = acc
models_dict['CNN'] = model_cnn
model_cnn.save('mnist_cnn.keras')
print(f"✓ Accuracy: {acc*100:.2f}% | Time: {training_times['CNN']:.1f}s")


# ===========================================================================
# METHOD 3: Improved CNN with Batch Normalization
# ===========================================================================
print("\n" + "-"*70)
print("METHOD 3: Improved CNN with Batch Normalization")
print("-"*70)

def create_improved_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

start_time = time.time()
model_improved = create_improved_cnn()
model_improved.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Improved CNN...")
history_improved = model_improved.fit(
    X_train_cnn, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

loss, acc = model_improved.evaluate(X_test_cnn, y_test, verbose=0)
training_times['Improved CNN'] = time.time() - start_time
results['Improved CNN'] = acc
models_dict['Improved CNN'] = model_improved
model_improved.save('mnist_improved_cnn.keras')
print(f"✓ Accuracy: {acc*100:.2f}% | Time: {training_times['Improved CNN']:.1f}s")


# ===========================================================================
# METHOD 4: Random Forest Classifier
# ===========================================================================
print("\n" + "-"*70)
print("METHOD 4: Random Forest Classifier")
print("-"*70)

start_time = time.time()
print("Training Random Forest (this may take a minute)...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
rf_clf.fit(X_train_norm, y_train)
y_pred_rf = rf_clf.predict(X_test_norm)
acc = accuracy_score(y_test, y_pred_rf)
training_times['Random Forest'] = time.time() - start_time
results['Random Forest'] = acc
models_dict['Random Forest'] = rf_clf
print(f"✓ Accuracy: {acc*100:.2f}% | Time: {training_times['Random Forest']:.1f}s")


# ===========================================================================
# RESULTS COMPARISON
# ===========================================================================
print("\n" + "="*70)
print("FINAL RESULTS - COMPARISON OF ALL METHODS")
print("="*70)
print(f"{'Method':<25} {'Accuracy':<15} {'Time (s)':<10} {'Status'}")
print("-"*70)

target_met = 0
for method in sorted(results.keys(), key=lambda x: results[x], reverse=True):
    status = "✓ TARGET MET" if results[method] >= 0.90 else "✗ Below 90%"
    if results[method] >= 0.90:
        target_met += 1
    print(f"{method:<25} {results[method]*100:>6.2f}%         {training_times[method]:>6.1f}     {status}")

best_method = max(results, key=results.get)
print("\n" + "="*70)
print(f" BEST MODEL: {best_method}")
print(f" BEST ACCURACY: {results[best_method]*100:.2f}%")
print(f" Models achieving >90%: {target_met}/{len(results)}")
print("="*70)


# ===========================================================================
# VISUALIZATIONS
# ===========================================================================
print("\nGenerating visualizations...")

# 1. Accuracy Comparison Bar Chart
plt.figure(figsize=(12, 6))
methods = list(results.keys())
accuracies = [results[m]*100 for m in methods]
colors = ['green' if acc >= 90 else 'orange' for acc in accuracies]

bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('MNIST Classification - Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90% Target')
plt.ylim(0, 100)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 1, 
             f'{acc:.2f}%', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('mnist_accuracy_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_accuracy_comparison.png")

# 2. Training Time Comparison
plt.figure(figsize=(12, 6))
times = [training_times[m] for m in methods]
plt.bar(methods, times, color='skyblue', edgecolor='black', linewidth=1.5)
plt.ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
plt.title('MNIST Classification - Training Time Comparison', fontsize=14, fontweight='bold')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', alpha=0.3)

for i, (method, t) in enumerate(zip(methods, times)):
    plt.text(i, t + max(times)*0.02, f'{t:.1f}s', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('mnist_time_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_time_comparison.png")

# 3. Confusion Matrix for Best Model
print(f"\nGenerating confusion matrix for {best_method}...")
if best_method == 'Random Forest':
    y_pred_best = y_pred_rf
else:
    best_model = models_dict[best_method]
    y_pred_best = np.argmax(best_model.predict(X_test_cnn, verbose=0), axis=1)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_method}\nAccuracy: {results[best_method]*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_confusion_matrix.png")

# 4. Per-digit Accuracy
print("\nCalculating per-digit accuracy...")
digit_accuracies = []
for digit in range(10):
    mask = y_test == digit
    digit_acc = np.mean(y_pred_best[mask] == y_test[mask]) * 100
    digit_accuracies.append(digit_acc)

plt.figure(figsize=(12, 6))
bars = plt.bar(range(10), digit_accuracies, color='lightgreen', edgecolor='black', linewidth=1.5)
plt.xlabel('Digit', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title(f'Per-Digit Accuracy - {best_method}', fontsize=14, fontweight='bold')
plt.xticks(range(10))
plt.ylim(0, 100)
plt.grid(axis='y', alpha=0.3)

for bar, acc in zip(bars, digit_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, acc + 1, 
             f'{acc:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('mnist_per_digit_accuracy.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_per_digit_accuracy.png")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\n Generated Files:")
print("  • mnist_simple_nn.keras")
print("  • mnist_cnn.keras")
print("  • mnist_improved_cnn.keras")
print("  • mnist_accuracy_comparison.png")
print("  • mnist_time_comparison.png")
print("  • mnist_confusion_matrix.png")
print("  • mnist_per_digit_accuracy.png")
print("\n ASSIGNMENT REQUIREMENT: Achieve >90% accuracy")
if target_met > 0:
    print(f"✓ STATUS: PASSED - {target_met} model(s) achieved >90% accuracy!")
else:
    print("✗ STATUS: FAILED - No models achieved >90% accuracy")
print("\n Next: Run q2_inference.py to visualize predictions!")
print("="*70)