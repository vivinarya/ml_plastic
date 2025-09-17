#!/usr/bin/env python3
"""
Improved training script for pixel-level microplastic classification.
Handles class imbalance by filtering rare classes.
"""

import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import load_cube, preprocess_cube, load_labels, extract_pixel_spectra, debug_data_sizes, create_classifier, apply_pca

def filter_rare_classes(X, y, min_samples_per_class=10):
    """Filter out classes with too few samples."""
    print(f"Filtering classes with < {min_samples_per_class} samples...")
    
    # Count samples per class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    print("Original class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    # Find classes with enough samples
    valid_classes = unique_classes[class_counts >= min_samples_per_class]
    
    print(f"\nKeeping {len(valid_classes)} classes with >= {min_samples_per_class} samples:")
    for cls in valid_classes:
        count = class_counts[unique_classes == cls][0]
        print(f"  Class {cls}: {count} samples")
    
    # Filter data
    if len(valid_classes) == 0:
        print("ERROR: No classes have enough samples!")
        return None, None
    
    # Create mask for valid samples
    valid_mask = np.isin(y, valid_classes)
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]
    
    # Remap class labels to be consecutive (0, 1, 2, ...)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes)}
    y_remapped = np.array([label_mapping[label] for label in y_filtered])
    
    print(f"\nAfter filtering:")
    print(f"  Samples: {len(X)} → {len(X_filtered)}")
    print(f"  Classes: {len(unique_classes)} → {len(valid_classes)}")
    print(f"  Labels remapped to: {list(range(len(valid_classes)))}")
    
    return X_filtered, y_remapped, label_mapping

def plot_class_distribution(y, title="Class Distribution"):
    """Plot class distribution."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_classes, class_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.xticks(unique_classes)
    for i, count in enumerate(class_counts):
        plt.text(unique_classes[i], count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(f'class_distribution_{title.lower().replace(" ", "_")}.png')
    plt.show()
    print(f"Class distribution plot saved as: class_distribution_{title.lower().replace(' ', '_')}.png")

def main():
    """Main training workflow with improved class handling."""
    
    # File paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    hdr_path = os.path.join(data_dir, 'microplastic.hdr')
    dat_path = os.path.join(data_dir, 'microplastic.dat')
    ilab_path = os.path.join(data_dir, 'microplastic.ilab')
    imsk_path = os.path.join(data_dir, 'microplastic.imsk')
    
    # Debug: Print all data sizes first
    print("=== DEBUGGING DATA SIZES ===")
    debug_data_sizes(hdr_path, imsk_path, ilab_path)
    print()
    
    # Check if files exist
    for path in [hdr_path, dat_path, ilab_path, imsk_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # 1. Load cube
    print("Loading cube...")
    cube = load_cube(hdr_path, dat_path)
    if cube is None:
        print("Failed to load cube. Exiting.")
        return
    
    print(f"Cube shape: {cube.shape}")
    
    # 2. Preprocess cube
    print("Preprocessing cube...")
    cube = preprocess_cube(cube)
    
    # 3. Load labels and mask
    print("Loading labels and mask...")
    labels, mask = load_labels(ilab_path, imsk_path, cube.shape)
    if labels is None or mask is None:
        print("Failed to load labels/mask. Exiting.")
        return
    
    # 4. Extract pixel spectra
    print("Extracting pixel spectra...")
    try:
        X = extract_pixel_spectra(cube, mask)
        y = labels
        
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        if len(X) != len(y):
            raise ValueError(f"Feature count {len(X)} != label count {len(y)}")
            
    except Exception as e:
        print(f"Error extracting spectra: {e}")
        return
    
    # 5. Filter rare classes (THIS IS THE KEY FIX!)
    print("\n" + "="*50)
    print("FILTERING RARE CLASSES")
    print("="*50)
    
    # Try different minimum thresholds
    min_samples = 15  # Require at least 15 samples per class
    result = filter_rare_classes(X, y, min_samples_per_class=min_samples)
    
    if result[0] is None:
        # Try with lower threshold
        print(f"\nNo classes with {min_samples}+ samples. Trying with 10...")
        min_samples = 10
        result = filter_rare_classes(X, y, min_samples_per_class=min_samples)
        
        if result[0] is None:
            # Try with even lower threshold
            print(f"\nNo classes with {min_samples}+ samples. Trying with 5...")
            min_samples = 5
            result = filter_rare_classes(X, y, min_samples_per_class=min_samples)
            
            if result[0] is None:
                print("ERROR: Cannot find any classes with enough samples!")
                return
    
    X_filtered, y_filtered, label_mapping = result
    
    # Plot class distribution
    plot_class_distribution(y_filtered, "Filtered Classes")
    
    # 6. Optional: Apply PCA for dimensionality reduction
    USE_PCA = True
    if USE_PCA:
        print("\nApplying PCA...")
        X_filtered, pca = apply_pca(X_filtered, n_components=100)  # Increased components
        print(f"After PCA: {X_filtered.shape}")
    else:
        pca = None
    
    # 7. Split data
    print("\nSplitting data...")
    # Now we can use stratify since we have enough samples per class
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.3, random_state=42, stratify=y_filtered
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Show class distribution in train/test
    print("\nTraining set class distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique_train, counts_train):
        print(f"  Class {cls}: {count} samples")
    
    # 8. Train classifier with better parameters
    print("\nTraining classifier...")
    clf = create_classifier(
        n_estimators=200,           # More trees
        max_depth=15,               # Prevent overfitting
        min_samples_split=5,        # Require more samples to split
        min_samples_leaf=2,         # Require more samples in leaves
        class_weight='balanced'     # Handle remaining imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # 9. Evaluate
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 10. Feature importance
    print("\nTop 10 Most Important Features:")
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        for i, feature_idx in enumerate(top_features):
            print(f"  {i+1}. Feature {feature_idx}: {importances[feature_idx]:.4f}")
    
    # 11. Save improved model
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'improved_pixel_classifier.pkl')
    
    # Save model with metadata
    model_data = {
        'classifier': clf,
        'pca': pca,
        'use_pca': USE_PCA,
        'label_mapping': label_mapping,
        'accuracy': accuracy,
        'min_samples_per_class': min_samples,
        'n_classes': len(np.unique(y_filtered)),
        'n_features_original': X.shape[1],
        'n_features_pca': X_filtered.shape[1] if USE_PCA else X.shape[1]
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nImproved model saved to: {model_path}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Classes used: {len(np.unique(y_filtered))}")
    print(f"Original classes: {len(np.unique(y))}")
    print(f"Samples used: {len(X_filtered)} / {len(X)}")
    print(f"Model saved: {model_path}")

if __name__ == "__main__":
    main()

