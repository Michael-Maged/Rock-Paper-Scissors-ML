import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Paths
MODELS_DIR = "models"

def resize_image(image, target_size=(100, 100)):
    """Resize image to target size"""
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
    Improved shape features:
    - fill ratio
    - number of connected components
    - largest component aspect ratio
    - edge density
    """
    from scipy import ndimage

    features = []

    gray = np.mean(image, axis=2)

    # adaptive threshold
    threshold = np.mean(gray)
    binary = (gray > threshold).astype(np.uint8)

    # --- 1) fill ratio ---
    fill_ratio = np.sum(binary) / binary.size
    features.append(fill_ratio)

    # --- 2) connected components ---
    labeled, num_components = ndimage.label(binary)
    features.append(num_components / 10.0)  # normalize

    # --- 3) largest component aspect ratio ---
    if num_components > 0:
        sizes = ndimage.sum(binary, labeled, range(1, num_components+1))
        largest = np.argmax(sizes) + 1
        coords = np.column_stack(np.where(labeled == largest))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h = max(1, y_max - y_min)
        w = max(1, x_max - x_min)
        aspect_ratio = w / h
    else:
        aspect_ratio = 1.0

    features.append(aspect_ratio)

    # --- 4) edge density (strong separator scissors vs rock) ---
    grad = np.hypot(np.diff(gray, axis=0, prepend=0), np.diff(gray, axis=1, prepend=0))
    edge_density = np.sum(grad > 25) / grad.size
    features.append(edge_density)

    return np.array(features)

def extract_all_features(image):
    """Extract all features from an image - MUST match training"""
    image_resized = resize_image(image, target_size=(100, 100))
    
    color_feat = extract_color_features(image_resized)
    edge_feat = extract_edge_features(image_resized)
    texture_feat = extract_texture_features(image_resized)
    shape_feat = extract_shape_features(image_resized)
    
    all_features = np.concatenate([color_feat, edge_feat, texture_feat, shape_feat])
    
    return all_features

def load_model():
    """Load the trained model"""
    print("="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train.py first.")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    selector = model_data['selector']
    class_names = model_data['class_names']
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Classes: {class_names}")
    
    return model, scaler, selector, class_names

def predict_image(image_path, model, scaler, selector, class_names):
    """Predict the class of a single image"""
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Extract features (MUST match training)
    features = extract_all_features(img_array)
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Apply feature selection if it was used
    if selector is not None:
        features_final = selector.transform(features_scaled)
    else:
        features_final = features_scaled
    
    # Predict
    prediction = model.predict(features_final)[0]
    predicted_class = class_names[prediction]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_final)[0]
    else:
        probabilities = None
    
    return predicted_class, probabilities, img

def visualize_prediction(image, predicted_class, probabilities, class_names):
    """Visualize prediction result"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f'Predicted: {predicted_class.upper()}',
                  fontsize=16, fontweight='bold', color='green')
    ax1.axis('off')
    
    # Display probabilities
    if probabilities is not None:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax2.barh(class_names, probabilities * 100, color=colors)
        ax2.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 100])
        
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, f'Predicted:\n{predicted_class.upper()}',
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_on_folder(folder_path, model, scaler, selector, class_names):
    """Test on all images in a folder"""
    print(f"\n{'='*70}")
    print(f"TESTING ON FOLDER: {folder_path}")
    print(f"{'='*70}")
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("No images found in folder!")
        return
    
    print(f"Found {len(image_files)} images\n")
    
    results = []
    correct = 0
    total = 0
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            predicted_class, probabilities, _ = predict_image(
                img_path, model, scaler, selector, class_names
            )
            
            # Try to extract true class from filename
            true_class = None
            for cls in class_names:
                if cls in img_file.lower():
                    true_class = cls
                    break
            
            if probabilities is not None:
                confidence = probabilities[class_names.index(predicted_class)] * 100
                result_str = f"{img_file}: {predicted_class} ({confidence:.1f}%)"
            else:
                result_str = f"{img_file}: {predicted_class}"
            
            if true_class:
                result_str += f" [True: {true_class}]"
                if true_class == predicted_class:
                    result_str += " ✓"
                    correct += 1
                else:
                    result_str += " ✗"
                total += 1
            
            print(result_str)
            results.append({
                'file': img_file,
                'predicted': predicted_class,
                'true': true_class,
                'correct': true_class == predicted_class if true_class else None
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n{'='*70}")
        print(f"ACCURACY: {accuracy:.2f}% ({correct}/{total})")
        print(f"{'='*70}")

def main():
    """Main function"""
    print("="*70)
    print("ROCK-PAPER-SCISSORS PREDICTION")
    print("PURE MACHINE LEARNING APPROACH")
    print("="*70)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    model, scaler, selector, class_names = model_data
    
    # Prediction options
    print(f"\n{'='*70}")
    print("PREDICTION OPTIONS")
    print("="*70)
    print("1. Predict single image")
    print("2. Test on folder")
    
    choice = input("\nSelect option (1/2): ").strip()
    
    if choice == '1':
        # Single image prediction
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
        
        try:
            predicted_class, probabilities, img = predict_image(
                image_path, model, scaler, selector, class_names
            )
            
            print(f"\n{'='*70}")
            print(f"PREDICTION RESULT")
            print(f"{'='*70}")
            print(f"Predicted class: {predicted_class.upper()}")
            
            if probabilities is not None:
                print("\nProbabilities:")
                for cls, prob in zip(class_names, probabilities):
                    print(f"  {cls}: {prob*100:.2f}%")
            print(f"{'='*70}")
            
            # Visualize
            visualize_prediction(img, predicted_class, probabilities, class_names)
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif choice == '2':
        # Test on folder
        folder_path = input("Enter folder path: ").strip()
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found at {folder_path}")
            return
        
        test_on_folder(folder_path, model, scaler, selector, class_names)
    
    print(f"\n{'='*70}")
    print("PREDICTION COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()