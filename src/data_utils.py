import numpy as np
import spectral
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def load_cube(hdr_path, dat_path):
    """Load spectral cube from ENVI format files."""
    try:
        cube = spectral.open_image(hdr_path)
        return cube.load()
    except Exception as e:
        print(f"Error loading cube: {e}")
        return None

def preprocess_cube(cube):
    """Normalize spectra in the cube."""
    print("Normalizing spectra...")
    
    # Convert to float to avoid overflow
    cube = cube.astype(np.float32)
    
    # Reshape to (pixels, bands)
    h, w, bands = cube.shape
    cube_reshaped = cube.reshape(-1, bands)
    
    # Normalize each spectrum (pixel) to unit length
    norms = np.linalg.norm(cube_reshaped, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    cube_normalized = cube_reshaped / norms
    
    # Reshape back to original shape
    return cube_normalized.reshape(h, w, bands)

def load_labels(ilab_path, imsk_path, cube_shape):
    """Load labels and mask, handling size mismatches."""
    print("Loading labels and mask...")
    
    # Load labels
    try:
        labels = np.fromfile(ilab_path, dtype=np.uint8)
        print(f"Labels loaded: {len(labels)} elements")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None, None
    
    # Load mask
    try:
        mask = np.fromfile(imsk_path, dtype=np.uint8)
        print(f"Mask loaded: {len(mask)} elements")
    except Exception as e:
        print(f"Error loading mask: {e}")
        return None, None
    
    # Expected spatial size from cube
    h, w, _ = cube_shape
    expected_size = h * w
    print(f"Expected spatial size from cube: {expected_size}")
    
    # Handle mask size mismatch
    if len(mask) != expected_size:
        print(f"[WARNING] Mask size {len(mask)} != expected {expected_size}")
        
        if len(mask) > expected_size:
            # Crop mask to match cube size
            print(f"Cropping mask from {len(mask)} to {expected_size}")
            mask = mask[:expected_size]
        else:
            # Pad mask if it's smaller
            print(f"Padding mask from {len(mask)} to {expected_size}")
            mask = np.pad(mask, (0, expected_size - len(mask)), mode='constant', constant_values=0)
    
    # Reshape mask to spatial dimensions
    mask = mask.reshape(h, w)
    
    # Handle labels size mismatch
    valid_pixels = np.sum(mask > 0)
    print(f"Valid pixels in mask: {valid_pixels}")
    
    if len(labels) != valid_pixels:
        print(f"[WARNING] Labels size {len(labels)} != valid pixels {valid_pixels}")
        
        if len(labels) > valid_pixels:
            # Truncate labels
            print(f"Truncating labels from {len(labels)} to {valid_pixels}")
            labels = labels[:valid_pixels]
        else:
            # Pad labels
            print(f"Padding labels from {len(labels)} to {valid_pixels}")
            labels = np.pad(labels, (0, valid_pixels - len(labels)), mode='constant', constant_values=0)
    
    return labels, mask

def extract_pixel_spectra(cube, mask):
    """Extract spectra for pixels where mask > 0."""
    print("Extracting spectra from valid pixels...")
    
    h, w, bands = cube.shape
    
    # Flatten cube to (pixels, bands)
    cube_flat = cube.reshape(-1, bands)
    
    # Flatten mask
    mask_flat = mask.flatten()
    
    # Check sizes match
    if len(cube_flat) != len(mask_flat):
        raise ValueError(f"Cube pixels {len(cube_flat)} != mask pixels {len(mask_flat)}")
    
    # Extract valid pixels
    valid_indices = mask_flat > 0
    valid_spectra = cube_flat[valid_indices]
    
    print(f"Extracted {valid_spectra.shape[0]} valid spectra with {valid_spectra.shape[1]} bands")
    return valid_spectra

def save_results(results, output_path):
    """Save classification results."""
    np.save(output_path, results)
    print(f"Results saved to {output_path}")

# Debug function to check data dimensions
def debug_data_sizes(cube_path, mask_path, labels_path):
    """Print all data dimensions for debugging."""
    print("=== DEBUG INFO ===")
    
    # Check cube
    try:
        cube = spectral.open_image(cube_path).load()
        print(f"Cube shape: {cube.shape}")
        print(f"Cube spatial size: {cube.shape[0] * cube.shape[1]}")
    except Exception as e:
        print(f"Cube error: {e}")
    
    # Check mask
    try:
        mask = np.fromfile(mask_path, dtype=np.uint8)
        print(f"Mask size: {len(mask)}")
    except Exception as e:
        print(f"Mask error: {e}")
    
    # Check labels
    try:
        labels = np.fromfile(labels_path, dtype=np.uint8)
        print(f"Labels size: {len(labels)}")
    except Exception as e:
        print(f"Labels error: {e}")
    
    print("==================")

def create_classifier(classifier_type='random_forest', **kwargs):
    """Create and return a classifier."""
    
    if classifier_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1),
            class_weight=kwargs.get('class_weight', 'balanced')
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

def apply_pca(X, n_components=None, variance_threshold=0.95):
    """Apply PCA for dimensionality reduction."""
    
    if n_components is None:
        # Determine number of components to retain variance_threshold of variance
        pca_temp = PCA()
        pca_temp.fit(X)
        
        cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= variance_threshold) + 1
        
        print(f"PCA: Using {n_components} components to retain {variance_threshold*100:.1f}% variance")
    
    # Apply PCA with determined number of components
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"PCA: Explained variance ratio: {explained_var:.4f}")
    
    return X_pca, pca

