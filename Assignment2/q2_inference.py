"""
CVI620 Assignment 2 - Q2: MNIST Classification
Inference and visualization script
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random

print("="*70)
print("MNIST DIGIT CLASSIFICATION - INFERENCE & VISUALIZATION")
print("="*70)

# Load test data
print("\nLoading MNIST test data...")
test_df = pd.read_csv('Q2/mnist_test.csv')
X_test = test_df.iloc[:, 1:].values / 255.0
y_test = test_df.iloc[:, 0].values
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

print(f"✓ Test samples: {len(X_test):,}")

# Load the best model
MODEL_PATH = 'mnist_improved_cnn.keras'
print(f"\nLoading best model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✓ Model loaded successfully!")

# Make predictions
print("\nMaking predictions on test set...")
predictions = model.predict(X_test_cnn, verbose=0)
y_pred = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"✓ Test Accuracy: {accuracy:.2f}%")

# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred))


# ===========================================================================
# VISUALIZATION 1: Random Sample Predictions
# ===========================================================================
def visualize_predictions(X, y_true, y_pred, predictions, num_samples=25):
    """Visualize random predictions with confidence scores"""
    
    print("\nGenerating random predictions visualization...")
    indices = random.sample(range(len(X)), num_samples)
    
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, sample_idx in enumerate(indices):
        img = X[sample_idx].reshape(28, 28)
        true_label = y_true[sample_idx]
        pred_label = y_pred[sample_idx]
        confidence = predictions[sample_idx][pred_label] * 100
        
        # Color: green if correct, red if wrong
        color = 'green' if pred_label == true_label else 'red'
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label} | Pred: {pred_label}\n{confidence:.1f}%',
                           color=color, fontsize=9, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Random Sample Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_sample_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mnist_sample_predictions.png")
    plt.close()

visualize_predictions(X_test, y_test, y_pred, predictions, num_samples=25)


# ===========================================================================
# VISUALIZATION 2: Misclassified Samples
# ===========================================================================
def visualize_errors(X, y_true, y_pred, predictions, num_samples=20):
    """Visualize misclassified samples"""
    
    error_indices = np.where(y_pred != y_true)[0]
    
    if len(error_indices) == 0:
        print("\n Perfect accuracy! No errors to display!")
        return
    
    print(f"\nTotal misclassifications: {len(error_indices)} ({len(error_indices)/len(y_true)*100:.2f}%)")
    print("Generating error visualization...")
    
    num_samples = min(num_samples, len(error_indices))
    selected_errors = random.sample(list(error_indices), num_samples)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, error_idx in enumerate(selected_errors):
        img = X[error_idx].reshape(28, 28)
        true_label = y_true[error_idx]
        pred_label = y_pred[error_idx]
        confidence = predictions[error_idx][pred_label] * 100
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label} | Pred: {pred_label}\n{confidence:.1f}%',
                           color='red', fontsize=9, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Misclassified Samples (Errors)', fontsize=14, fontweight='bold', color='red')
    plt.tight_layout()
    plt.savefig('mnist_errors.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mnist_errors.png")
    plt.close()

visualize_errors(X_test, y_test, y_pred, predictions, num_samples=20)


# ===========================================================================
# VISUALIZATION 3: One Sample of Each Digit (0-9)
# ===========================================================================
def visualize_each_digit(X, y_true, y_pred, predictions):
    """Show one correctly predicted sample for each digit"""
    
    print("\nGenerating visualization of each digit (0-9)...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        # Find first correct prediction of this digit
        correct_indices = np.where((y_true == digit) & (y_pred == digit))[0]
        
        if len(correct_indices) > 0:
            idx = correct_indices[0]
            img = X[idx].reshape(28, 28)
            confidence = predictions[idx][digit] * 100
            
            axes[digit].imshow(img, cmap='gray')
            axes[digit].set_title(f'Digit: {digit}\nConf: {confidence:.1f}%',
                                 fontsize=10, fontweight='bold', color='green')
            axes[digit].axis('off')
    
    plt.suptitle('Sample of Each Digit (0-9)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mnist_all_digits.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mnist_all_digits.png")
    plt.close()

visualize_each_digit(X_test, y_test, y_pred, predictions)


# ===========================================================================
# VISUALIZATION 4: Confidence Distribution
# ===========================================================================
def visualize_confidence_analysis(predictions, y_true, y_pred):
    """Analyze prediction confidence"""
    
    print("\nGenerating confidence analysis...")
    
    confidences = np.max(predictions, axis=1)
    correct_mask = y_pred == y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence histogram
    axes[0].hist(confidences[correct_mask], bins=50, alpha=0.7, 
                label='Correct', color='green', edgecolor='black')
    axes[0].hist(confidences[~correct_mask], bins=50, alpha=0.7, 
                label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-digit accuracy
    digit_accuracies = []
    for digit in range(10):
        mask = y_true == digit
        digit_acc = np.mean(y_pred[mask] == y_true[mask]) * 100
        digit_accuracies.append(digit_acc)
    
    bars = axes[1].bar(range(10), digit_accuracies, color='skyblue', 
                       edgecolor='navy', linewidth=1.5)
    axes[1].set_xlabel('Digit', fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Per-Digit Accuracy', fontweight='bold')
    axes[1].set_xticks(range(10))
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, digit_accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, acc + 1, 
                    f'{acc:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('mnist_confidence_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: mnist_confidence_analysis.png")
    plt.close()

visualize_confidence_analysis(predictions, y_test, y_pred)


# ===========================================================================
# SUMMARY
# ===========================================================================
print("\n" + "="*70)
print("INFERENCE COMPLETE - SUMMARY")
print("="*70)
print(f"Total test samples: {len(y_test):,}")
print(f"Correct predictions: {np.sum(y_pred == y_test):,}")
print(f"Incorrect predictions: {np.sum(y_pred != y_test):,}")
print(f"Final Accuracy: {accuracy:.2f}%")

# Find best and worst performing digits
digit_accuracies = []
for digit in range(10):
    mask = y_test == digit
    digit_acc = np.mean(y_pred[mask] == y_test[mask]) * 100
    digit_accuracies.append((digit, digit_acc))

digit_accuracies.sort(key=lambda x: x[1], reverse=True)
print(f"\nBest performing digit: {digit_accuracies[0][0]} ({digit_accuracies[0][1]:.2f}%)")
print(f"Worst performing digit: {digit_accuracies[-1][0]} ({digit_accuracies[-1][1]:.2f}%)")

print("\n" + "="*70)
print(" Generated Visualization Files:")
print("="*70)
print("  • mnist_sample_predictions.png")
print("  • mnist_errors.png")
print("  • mnist_all_digits.png")
print("  • mnist_confidence_analysis.png")
print("\n All files ready for your assignment submission!")
print("="*70)