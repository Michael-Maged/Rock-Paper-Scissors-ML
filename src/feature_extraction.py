import os
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import cv2

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

def segment_hand(image):
    """Isolate hand from background using skin color segmentation"""
    # Convert to YCrCb for better skin detection
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Skin color thresholds in YCrCb
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(ycrcb, lower, upper)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def get_hand_contour(mask):
    """Find the largest contour (hand)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)

def extract_convex_hull_features(contour):
    """Extract convex hull and convexity defects"""
    if contour is None or len(contour) < 10:
        return [0, 0, 0]
    
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 4:
        return [0, 0, 0]
    
    defects = cv2.convexityDefects(contour, hull)
    
    if defects is None:
        return [0, 0, 0]
    
    # Count significant defects (deep valleys between fingers)
    significant_defects = 0
    total_depth = 0
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0  # Convert to actual distance
        if depth > 10:  # Threshold for significant defect
            significant_defects += 1
            total_depth += depth
    
    avg_depth = total_depth / max(1, significant_defects)
    
    return [significant_defects, avg_depth, len(hull)]

def extract_bounding_box_features(contour):
    """Extract bounding box aspect ratio"""
    if contour is None:
        return [1.0, 0]
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(1, h)
    area = cv2.contourArea(contour)
    bbox_area = w * h
    extent = area / max(1, bbox_area)
    
    return [aspect_ratio, extent]

def extract_area_ratio_features(contour):
    """Extract area ratios (solidity)"""
    if contour is None:
        return [0, 0]
    
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    solidity = area / max(1, hull_area)
    perimeter = cv2.arcLength(contour, True)
    compactness = (4 * np.pi * area) / max(1, perimeter ** 2)
    
    return [solidity, compactness]

def extract_centroid_distance_profile(contour):
    """Extract distance profile from centroid"""
    if contour is None or len(contour) < 10:
        return [0, 0, 0]
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return [0, 0, 0]
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # Calculate distances from centroid to contour points
    distances = []
    for point in contour:
        x, y = point[0]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Find peaks (potential fingertips)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    peaks = np.sum(distances > (mean_dist + 0.5 * std_dist))
    
    return [np.mean(distances), std_dist, peaks]

def extract_hu_moments(contour):
    """Extract Hu moments (shape descriptors)"""
    if contour is None:
        return [0] * 7
    
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    
    # Log transform to make them more manageable
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments.flatten().tolist()

def count_fingers(contour):
    """Count extended fingers using k-curvature"""
    if contour is None or len(contour) < 20:
        return 0
    
    # Simplify contour
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 6:
        return 0
    
    # Find convex hull and defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 4:
        return 0
    
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    
    # Count fingertips (points between defects)
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 8000:  # Significant defect depth
            finger_count += 1
    
    return min(finger_count, 5)  # Max 5 fingers

def extract_fist_detection_features(image):
    """Specific features to detect closed fist (rock)"""
    mask = segment_hand(image)
    
    if mask is None:
        return [0, 0, 0]
    
    # 1. Compactness (rock should be very compact)
    contour = get_hand_contour(mask)
    if contour is None:
        return [0, 0, 0]
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return [0, 0, 0]
    
    # Circularity (rock is more circular)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # 2. Aspect ratio of bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(h, 1)
    
    # 3. Extent (how much of bounding box is filled)
    rect_area = w * h
    extent = area / max(rect_area, 1)
    
    return [circularity, aspect_ratio, extent]

def extract_geometric_features(image):
    """Extract all geometric features from hand segmentation"""
    features = []
    
    # Segment hand
    mask = segment_hand(image)
    contour = get_hand_contour(mask)
    
    # Extract features
    convex_features = extract_convex_hull_features(contour)
    bbox_features = extract_bounding_box_features(contour)
    area_features = extract_area_ratio_features(contour)
    distance_features = extract_centroid_distance_profile(contour)
    hu_features = extract_hu_moments(contour)
    finger_count = count_fingers(contour)
    fist_features = extract_fist_detection_features(image)
    
    # Combine all geometric features
    features.extend(convex_features)  # 3 features
    features.extend(bbox_features)    # 2 features
    features.extend(area_features)    # 2 features
    features.extend(distance_features) # 3 features
    features.extend(hu_features)      # 7 features
    features.append(finger_count)     # 1 feature
    features.extend(fist_features)    # 3 features
    
    return np.array(features)  # Total: 18 features

def extract_finger_counting_features(image):
    """Better finger detection"""
    mask = segment_hand(image)
    contour = get_hand_contour(mask)
    
    if contour is None or len(contour) < 10:
        return [0, 0, 0, 0, 0]
    
    # Find convex hull
    hull = cv2.convexHull(contour, returnPoints=False)
    
    if len(hull) < 4:
        return [0, 0, 0, 0, 0]
    
    # Get defects
    defects = cv2.convexityDefects(contour, hull)
    
    if defects is None:
        return [0, 0, 0, 0, 0]
    
    # Count significant defects (valleys between fingers)
    significant_defects = 0
    max_depth = 0
    avg_depth = 0
    depths = []
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        
        # Get the points
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # Calculate angle at the defect point
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b + 1e-5))
        angle_deg = np.degrees(angle)
        
        # Count only defects with reasonable depth and angle
        if depth > 15 and angle_deg < 90:  # Deep valley with acute angle
            significant_defects += 1
            depths.append(depth)
            max_depth = max(max_depth, depth)
    
    if depths:
        avg_depth = np.mean(depths)
    
    # Estimate finger count (defects + 1)
    estimated_fingers = min(significant_defects + 1, 5)
    
    return [
        estimated_fingers,
        significant_defects,
        max_depth,
        avg_depth,
        len(depths)
    ]

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
    geometric_feat = extract_geometric_features(image_resized)
    finger_features = extract_finger_counting_features(image_resized)
    
    # Combine all features
    all_features = np.concatenate([geometric_feat, finger_features])
    
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
    print(f"  Geometric features: 18 (hand segmentation + shape analysis)")
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