import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
LABEL_MAP = {"rock": 0, "paper": 1, "scissors": 2}

def load_images_and_labels(split="training"):
    images, labels = [], []
    
    if split == "validation":
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split
            for img_path in path.glob(f"{gesture}*.png"):
                img = Image.open(img_path)
                images.append(np.array(img))
                labels.append(label)
    else:
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split / gesture
            for img_path in path.glob("*.png"):
                img = Image.open(img_path)
                images.append(np.array(img))
                labels.append(label)
    
    return np.array(images), np.array(labels)

def extract_handcrafted_features(split="training", target_size=(224, 224)):
    from skimage.feature import hog
    from skimage.filters import sobel
    from skimage.transform import resize
    
    print(f"\nExtracting features from {split} set...")
    images, labels = load_images_and_labels(split)
    features_list = []
    
    for idx, img in enumerate(images):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(images)}")
        
        # Resize and normalize
        img_resized = resize(img, target_size, anti_aliasing=True)
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
            img_resized = img_resized[:, :, :3]
        
        # Grayscale
        gray = np.mean(img_resized, axis=2) if len(img_resized.shape) == 3 else img_resized
        
        # HOG features
        hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        
        # Color histogram
        if len(img_resized.shape) == 3:
            hist = [np.histogram(img_resized[:,:,i], bins=32, range=(0, 1))[0] for i in range(3)]
            color_hist = np.concatenate(hist)
        else:
            color_hist = np.histogram(img_resized, bins=32, range=(0, 1))[0]
        
        # Edge density
        edges = sobel(gray)
        edge_density = np.sum(edges > edges.mean()) / edges.size
        
        combined = np.concatenate([hog_feat, color_hist, [edge_density]])
        features_list.append(combined)
    
    features = np.array(features_list)
    print(f"  Feature shape: {features.shape}")
    return features, labels