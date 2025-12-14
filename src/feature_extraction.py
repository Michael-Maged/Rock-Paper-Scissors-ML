from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.filters import sobel
from skimage.transform import resize
import pickle
import os
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from explore_data import load_images_and_labels

def extract_features_handcrafted(split="training", target_size=(224, 224)):
    print(f"\n{'='*70}")
    print(f"EXTRACTING HANDCRAFTED FEATURES FROM {split.upper()} SET")
    print("="*70)
    
    images, labels = load_images_and_labels(split)
    print(f"Loaded {len(images)} images")
    
    features_list = []
    
    for idx, img in enumerate(images):
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(images)}")
        
        # Resize and normalize
        img_resized = resize(img, target_size, anti_aliasing=True)
        
        # Handle RGBA images (convert to RGB)
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
            img_resized = img_resized[:, :, :3]
        
        # Convert to grayscale for some features
        if len(img_resized.shape) == 3:
            gray = np.mean(img_resized, axis=2)
        else:
            gray = img_resized
        
        # 1. HOG Features
        try:
            hog_feat = hog(
                gray, 
                pixels_per_cell=(16, 16), 
                cells_per_block=(2, 2), 
                feature_vector=True
            )
        except Exception as e:
            print(f"Warning: HOG extraction failed for image {idx}: {e}")
            hog_feat = np.zeros(1296)  # Default HOG size
        
        # 2. Color Histogram
        if len(img_resized.shape) == 3:
            hist = []
            for i in range(3):  # RGB channels
                h, _ = np.histogram(img_resized[:,:,i], bins=32, range=(0, 1))
                hist.append(h)
            color_hist = np.concatenate(hist)
        else:
            color_hist, _ = np.histogram(img_resized, bins=32, range=(0, 1))
        
        # 3. Edge Features (Edge Density)
        edges = sobel(gray)
        edge_density = np.sum(edges > edges.mean()) / edges.size
        
        # 4. Statistical Features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Combine all features
        combined = np.concatenate([
            hog_feat, 
            color_hist, 
            [edge_density, mean_val, std_val]
        ])
        
        features_list.append(combined)
    
    features = np.array(features_list)
    print(f"  Feature shape: {features.shape}")
    print(f"  Feature dimensionality: {features.shape[1]}")
    
    return features, labels

def extract_features_cnn(split="training", target_size=(224, 224), model_name='MobileNetV2'):
    if VGG16 is None:
        raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING DEEP FEATURES ({model_name}) FROM {split.upper()} SET")
    print("="*70)
    
    # Load images
    images, labels = load_images_and_labels(split)
    print(f"Loaded {len(images)} images")
    
    # Prepare images
    processed_images = []
    for img in images:
        # Resize
        img_resized = resize(img, target_size, anti_aliasing=True)
        
        # Handle RGBA
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
            img_resized = img_resized[:, :, :3]
        
        # Ensure 3 channels (RGB)
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        # Scale to 0-255
        img_resized = (img_resized * 255).astype(np.uint8)
        
        processed_images.append(img_resized)
    
    processed_images = np.array(processed_images)
    print(f"Processed images shape: {processed_images.shape}")
    
    # Load pre-trained model
    print(f"Loading {model_name} model...")
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(224, 224, 3))
        preprocess_fn = vgg_preprocess
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(224, 224, 3))
        preprocess_fn = resnet_preprocess
    else:  # MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(224, 224, 3))
        preprocess_fn = mobilenet_preprocess
    
    # Preprocess
    processed_images = preprocess_fn(processed_images.astype(float))
    
    # Extract features
    print("Extracting features... This may take a few minutes.")
    features = base_model.predict(processed_images, batch_size=32, verbose=1)
    
    # Flatten features
    features_flat = features.reshape(features.shape[0], -1)
    
    print(f"Feature shape: {features_flat.shape}")
    print(f"Feature dimensionality: {features_flat.shape[1]}")
    
    return features_flat, labels

def save_features(features, labels, filename):
    os.makedirs('data/features', exist_ok=True)
    filepath = f'data/features/{filename}'
    
    data = {
        'features': features,
        'labels': labels
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Features saved to: {filepath}")

if __name__ == "__main__":
    import os
    os.makedirs('data/features', exist_ok=True)
    
    splits = ['training', 'validation', 'testing']
    
    print("="*70)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*70)
    print("\nChoose feature extraction method:")
    print("1. Handcrafted features (HOG, Color Histograms, Edge features)")
    print("2. Deep features (MobileNetV2 - recommended)")
    print("3. Both (takes longer)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"PROCESSING {split.upper()} SET")
        print("="*70)
        
        if choice in ['1', '3']:
            # Extract handcrafted features
            features_hc, labels = extract_features_handcrafted(split)
            save_features(features_hc, labels, f'{split}_handcrafted.pkl')
        
        if choice in ['2', '3']:
            # Extract deep features
            features_deep, labels = extract_features_cnn(split, model_name='MobileNetV2')
            save_features(features_deep, labels, f'{split}_deep_mobilenet.pkl')
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETED!")
    print("="*70)
    print("\nExtracted features saved in: data/features/")
    print("\nNext step: Run training and classification script")