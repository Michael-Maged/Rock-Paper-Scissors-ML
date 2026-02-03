import os
import numpy as np
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
import cv2

# Try importing MediaPipe (handles both old and new API)
try:
    import mediapipe as mp
    
    # Force use of old API (simpler, no model file needed)
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        MEDIAPIPE_API = 'old'
        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe loaded successfully!")
    except AttributeError:
        # If old API not available, try new API
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Download model if not exists
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        hands = vision.HandLandmarker.create_from_options(options)
        MEDIAPIPE_API = 'new'
        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe loaded successfully (NEW API)!")
        
except (ImportError, AttributeError) as e:
    print(f"Warning: MediaPipe not available - {e}")
    print("Please install MediaPipe: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_API = None
    hands = None

# Paths
DATA_DIR = "data"  # Changed to use training folder
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
                images.append((np.array(img), filename))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return images

def extract_hand_landmarks(image):
    """
    Extract 21 hand landmarks using MediaPipe
    Returns normalized (x, y, z) coordinates for each landmark
    Works with both old and new MediaPipe API
    """
    # Convert to RGB if needed (MediaPipe expects RGB)
    if len(image.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image
    
    if MEDIAPIPE_API == 'new':
        # New API (MediaPipe 0.10.30+)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = hands.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), hand_landmarks
    else:
        # Old API (MediaPipe < 0.10.30)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), hand_landmarks
    
    return None, None

def calculate_relative_distances(hand_landmarks):
    """
    Calculate relative distances between key landmarks
    Works with both old and new MediaPipe API
    
    Landmark indices:
    0: Wrist
    4: Thumb tip
    8: Index finger tip
    12: Middle finger tip
    16: Ring finger tip
    20: Pinky tip
    """
    if hand_landmarks is None:
        return np.zeros(15)  # Return zeros if no hand detected
    
    # Extract landmark coordinates (works for both list and object)
    def get_coords(idx):
        if MEDIAPIPE_API == 'new':
            # New API: landmarks is a list
            lm = hand_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])
        else:
            # Old API: landmarks.landmark is the list
            lm = hand_landmarks.landmark[idx]
            return np.array([lm.x, lm.y, lm.z])
    
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    features = []
    
    # Key landmark indices
    wrist = get_coords(0)
    thumb_tip = get_coords(4)
    index_tip = get_coords(8)
    middle_tip = get_coords(12)
    ring_tip = get_coords(16)
    pinky_tip = get_coords(20)
    
    # Distance from each fingertip to wrist
    features.append(euclidean_distance(thumb_tip, wrist))
    features.append(euclidean_distance(index_tip, wrist))
    features.append(euclidean_distance(middle_tip, wrist))
    features.append(euclidean_distance(ring_tip, wrist))
    features.append(euclidean_distance(pinky_tip, wrist))
    
    # Distance between fingertips (especially useful for scissors)
    features.append(euclidean_distance(thumb_tip, index_tip))
    features.append(euclidean_distance(index_tip, middle_tip))  # Key for scissors
    features.append(euclidean_distance(middle_tip, ring_tip))
    features.append(euclidean_distance(ring_tip, pinky_tip))
    features.append(euclidean_distance(thumb_tip, pinky_tip))
    
    # Distance from fingertips to palm center (landmark 0, 5, 9, 13, 17 avg)
    palm_center = (get_coords(0) + get_coords(5) + get_coords(9) + 
                   get_coords(13) + get_coords(17)) / 5
    
    features.append(euclidean_distance(thumb_tip, palm_center))
    features.append(euclidean_distance(index_tip, palm_center))
    features.append(euclidean_distance(middle_tip, palm_center))
    features.append(euclidean_distance(ring_tip, palm_center))
    features.append(euclidean_distance(pinky_tip, palm_center))
    
    return np.array(features)

def calculate_finger_angles(hand_landmarks):
    """
    Calculate angles of fingers relative to palm
    Useful for detecting extended vs folded fingers
    Works with both old and new MediaPipe API
    """
    if hand_landmarks is None:
        return np.zeros(5)
    
    def get_coords(idx):
        if MEDIAPIPE_API == 'new':
            lm = hand_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])
        else:
            lm = hand_landmarks.landmark[idx]
            return np.array([lm.x, lm.y, lm.z])
    
    def calculate_angle(p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle
    
    angles = []
    
    # Finger base, middle, tip indices
    fingers = [
        (1, 2, 4),   # Thumb
        (5, 6, 8),   # Index
        (9, 10, 12), # Middle
        (13, 14, 16),# Ring
        (17, 18, 20) # Pinky
    ]
    
    for base, mid, tip in fingers:
        angle = calculate_angle(get_coords(base), get_coords(mid), get_coords(tip))
        angles.append(angle)
    
    return np.array(angles)

def extract_all_features(image):
    """
    Extract all hand-based features from an image
    """
    # Extract hand landmarks
    landmarks_flat, hand_landmarks = extract_hand_landmarks(image)
    
    if landmarks_flat is None:
        # Return zeros if no hand detected
        return None
    
    # Calculate relative distances
    distances = calculate_relative_distances(hand_landmarks)
    
    # Calculate finger angles
    angles = calculate_finger_angles(hand_landmarks)
    
    # Combine all features
    all_features = np.concatenate([landmarks_flat, distances, angles])
    
    return all_features

def extract_features_from_dataset():
    """
    Extract features from all images in the dataset
    """
    print("="*70)
    print("HAND LANDMARK FEATURE EXTRACTION - MediaPipe")
    print("="*70)
    print(f"Reading from: {DATA_DIR}")
    
    X = []  # Features
    y = []  # Labels
    failed_images = []
    
    classes = ['rock', 'paper', 'scissors']
    
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found!")
            continue
        
        print(f"\nProcessing {class_name}...")
        images = load_images_from_folder(class_path)
        print(f"  Found {len(images)} images")
        
        if len(images) == 0:
            print(f"  ERROR: No images found in {class_path}")
            continue
        
        processed = 0
        for idx, (image, filename) in enumerate(images):
            if idx % 50 == 0 and idx > 0:
                print(f"  Processed {idx}/{len(images)} images")
            
            try:
                features = extract_all_features(image)
                if features is not None:
                    X.append(features)
                    y.append(CLASS_MAP[class_name])
                    processed += 1
                else:
                    failed_images.append((class_name, filename))
            except Exception as e:
                print(f"  Error processing image {idx} ({filename}): {e}")
                failed_images.append((class_name, filename))
        
        print(f"  Successfully processed: {processed}/{len(images)} images")
    
    if len(X) == 0:
        print("\n" + "="*70)
        print("ERROR: No features extracted!")
        print("="*70)
        return np.array([]), np.array([])
    
    X = np.array(X)
    y = np.array(y)
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETED")
    print("="*70)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")
    print(f"Failed to detect hands: {len(failed_images)}")
    
    if failed_images and len(failed_images) < 20:
        print("\nFailed images:")
        for class_name, filename in failed_images[:10]:
            print(f"  {class_name}/{filename}")
    
    # Feature breakdown
    print("\nFeature composition:")
    print("  Hand landmarks (21 × 3): 63 features (x, y, z coordinates)")
    print("  Relative distances: 15 features (fingertip distances)")
    print("  Finger angles: 5 features (finger bend angles)")
    print(f"  Total: {X.shape[1]} features")
    
    return X, y

def split_and_save_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets and save
    """
    print("\n" + "="*70)
    print("SPLITTING DATASET")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Print class distribution
    for class_name, class_id in CLASS_MAP.items():
        train_count = np.sum(y_train == class_id)
        test_count = np.sum(y_test == class_id)
        print(f"  {class_name}: {train_count} train, {test_count} test")
    
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
    print("  - train_features.pkl")
    print("  - test_features.pkl")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function"""
    # Check MediaPipe availability
    if not MEDIAPIPE_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: MediaPipe is not properly installed!")
        print("="*70)
        print("\nPlease install MediaPipe using ONE of these commands:")
        print("\nOption 1 (Recommended - latest stable):")
        print("  pip install mediapipe")
        print("\nOption 2 (Specific working version):")
        print("  pip install mediapipe==0.10.14")
        print("\nOption 3 (If using conda):")
        print("  conda install -c conda-forge mediapipe")
        print("\nAfter installation, run this script again.")
        print("="*70)
        return
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print("\nError: Data directory '{DATA_DIR}' not found!")
        print("Expected structure:")
        print("  data/")
        print("  ├── training/")
        print("  │   ├── rock/")
        print("  │   ├── paper/")
        print("  │   └── scissors/")
        print("  └── testing/")
        print("      ├── rock/")
        print("      ├── paper/")
        print("      └── scissors/")
        return
    
    # Extract features
    X, y = extract_features_from_dataset()
    
    if len(X) == 0:
        print("\nError: No features extracted!")
        print("Make sure your images contain visible hands.")
        return
    
    # Split and save
    X_train, X_test, y_train, y_test = split_and_save_data(X, y)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Feature extraction completed successfully!")
    print(f"Total features per image: {X.shape[1]}")
    print("Feature types:")
    print("  - 63 hand landmark coordinates (x, y, z for 21 points)")
    print("  - 15 relative distances between key points")
    print("  - 5 finger angles")
    print("\nThese features capture:")
    print("  ✓ Hand pose and finger positions")
    print("  ✓ Finger spread (important for scissors)")
    print("  ✓ Fist closure (important for rock)")
    print("  ✓ Open palm (important for paper)")
    print("\nReady for training!")
    print("\nNext step: Run train.py")
    print("="*70)

if __name__ == "__main__":
    main()