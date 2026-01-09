import os
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "data"
FEATURES_DIR = "features"

# Create features directory
os.makedirs(FEATURES_DIR, exist_ok=True)

# Class mapping
CLASS_MAP = {'rock': 0, 'paper': 1, 'scissors': 2}

def load_images_from_folder(folder_path):
    """Load all images from a folder"""
    images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(np.array(img))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return images

def resize_image(image, target_size=(100, 100)):
    """Resize image to target size using PIL"""
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize(target_size, Image.BILINEAR)
    return np.array(img_resized)

def extract_color_features(image):
    """
    Extract color-based features
    - Mean, std, min, max for each RGB channel
    - Color histograms
    """
    features = []
    
    # Separate RGB channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    # Statistical features for each channel
    for channel in [r_channel, g_channel, b_channel]:
        features.append(np.mean(channel))
        features.append(np.std(channel))
        features.append(np.min(channel))
        features.append(np.max(channel))
        features.append(np.median(channel))
    
    # Color histograms (16 bins per channel)
    for channel in [r_channel, g_channel, b_channel]:
        hist, _ = np.histogram(channel.flatten(), bins=16, range=(0, 256))
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist)
    
    return np.array(features)

def extract_edge_features(image):
    """
    Extract simple edge features using gradients
    """
    features = []
    
    # Convert to grayscale
    gray = np.mean(image, axis=2)
    
    # Compute horizontal and vertical gradients (simple difference)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    
    # Edge statistics
    features.append(np.mean(grad_x))
    features.append(np.std(grad_x))
    features.append(np.max(grad_x))
    
    features.append(np.mean(grad_y))
    features.append(np.std(grad_y))
    features.append(np.max(grad_y))
    
    # Edge density (pixels with high gradient)
    threshold = 30
    edge_pixels_x = np.sum(grad_x > threshold)
    edge_pixels_y = np.sum(grad_y > threshold)
    features.append(edge_pixels_x / grad_x.size)
    features.append(edge_pixels_y / grad_y.size)
    
    return np.array(features)

def extract_texture_features(image):
    """
    Extract simple texture features
    """
    features = []
    
    # Convert to grayscale
    gray = np.mean(image, axis=2)
    
    # Compute local variance (texture measure)
    # Using 5x5 patches
    h, w = gray.shape
    patch_size = 5
    variances = []
    
    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            patch = gray[i:i+patch_size, j:j+patch_size]
            variances.append(np.var(patch))
    
    features.append(np.mean(variances))
    features.append(np.std(variances))
    features.append(np.max(variances))
    features.append(np.min(variances))
    
    # Image entropy (measure of randomness)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-7)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    features.append(entropy)
    
    return np.array(features)

def extract_shape_features(image):
    """
    Extract simple shape-related features
    """
    features = []
    
    # Convert to grayscale
    gray = np.mean(image, axis=2)
    
    # Binarize (simple thresholding)
    threshold = np.mean(gray)
    binary = (gray > threshold).astype(np.uint8)
    
    # Shape statistics
    features.append(np.sum(binary) / binary.size)  # Fill ratio
    
    # Moments (simple center of mass)
    y_coords, x_coords = np.where(binary > 0)
    if len(x_coords) > 0:
        center_x = np.mean(x_coords) / binary.shape[1]
        center_y = np.mean(y_coords) / binary.shape[0]
        features.append(center_x)
        features.append(center_y)
    else:
        features.extend([0.5, 0.5])
    
    # Spread (variance of coordinates)
    if len(x_coords) > 0:
        features.append(np.std(x_coords) / binary.shape[1])
        features.append(np.std(y_coords) / binary.shape[0])
    else:
        features.extend([0, 0])
    
    return np.array(features)

def extract_all_features(image):
    """
    Extract all features from an image
    """
    # Resize image for consistency
    image_resized = resize_image(image, target_size=(100, 100))
    
    # Extract different feature types
    color_feat = extract_color_features(image_resized)
    edge_feat = extract_edge_features(image_resized)
    texture_feat = extract_texture_features(image_resized)
    shape_feat = extract_shape_features(image_resized)
    
    # Combine all features
    all_features = np.concatenate([color_feat, edge_feat, texture_feat, shape_feat])
    
    return all_features

def extract_features_from_dataset():
    """
    Extract features from all images in the dataset
    """
    print("="*70)
    print("FEATURE EXTRACTION - PURE MACHINE LEARNING")
    print("="*70)
    
    X = []  # Features
    y = []  # Labels
    
    classes = ['rock', 'paper', 'scissors']
    
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found!")
            continue
        
        print(f"\nProcessing {class_name}...")
        images = load_images_from_folder(class_path)
        print(f"  Found {len(images)} images")
        
        for idx, image in enumerate(images):
            if idx % 50 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(images)} images")
            
            try:
                features = extract_all_features(image)
                X.append(features)
                y.append(CLASS_MAP[class_name])
            except Exception as e:
                print(f"  Error processing image {idx}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n" + "="*70)
    print(f"FEATURE EXTRACTION COMPLETED")
    print(f"="*70)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")
    
    # Feature breakdown
    print(f"\nFeature composition:")
    print(f"  Color features: 63 (RGB stats + histograms)")
    print(f"  Edge features: 8 (gradient statistics)")
    print(f"  Texture features: 5 (variance + entropy)")
    print(f"  Shape features: 5 (fill ratio + moments)")
    print(f"  Total: {X.shape[1]} features")
    
    return X, y

def split_and_save_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets and save
    """
    print(f"\n" + "="*70)
    print(f"SPLITTING DATASET")
    print(f"="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Save features
    train_data = {
        'X_train': X_train,
        'y_train': y_train
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    
    with open(os.path.join(FEATURES_DIR, 'train_features.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(FEATURES_DIR, 'test_features.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    print(f"\nFeatures saved to: {FEATURES_DIR}/")
    print(f"  - train_features.pkl")
    print(f"  - test_features.pkl")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function"""
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        return
    
    # Extract features
    X, y = extract_features_from_dataset()
    
    if len(X) == 0:
        print("Error: No features extracted!")
        return
    
    # Split and save
    X_train, X_test, y_train, y_test = split_and_save_data(X, y)
    
    # Summary
    print(f"\n" + "="*70)
    print(f"SUMMARY")
    print(f"="*70)
    print(f"Feature extraction completed successfully!")
    print(f"Total features per image: {X.shape[1]}")
    print(f"Ready for training!")
    print(f"\nNext step: Run train.py")
    print(f"="*70)

if __name__ == "__main__":
    main()