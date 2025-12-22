import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import sobel
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from keras.applications import MobileNetV2
    from keras.applications.mobilenet_v2 import preprocess_input
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from explore_data import load_images_and_labels

# Cache for deep learning model to avoid reloading
base_model_cache = None
    
def load_model(model_path='models/best_model_handcrafted.pkl'):
    print("="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    scaler = data['scaler']
    pca = data['pca']
    class_names = data['class_names']
    feature_type = data['feature_type']
    
    print(f"Model loaded: {model_path}")
    print(f"Model type: {type(model).__name__}")
    print(f"Feature type: {feature_type}")
    print(f"Classes: {class_names}")
    
    return model, scaler, pca, class_names, feature_type

def extract_features_from_image(img_array, feature_type='handcrafted', target_size=(224, 224)):
    """
    Extract features from a single image
    CRITICAL: This must match EXACTLY the feature extraction in training
    """
    # Resize to target size and normalize to [0, 1]
    img_resized = resize(img_array, target_size, anti_aliasing=True)
    
    # Handle RGBA images (convert to RGB)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]
    
    if feature_type == 'handcrafted':
        # === HANDCRAFTED FEATURES ===
        # Convert to grayscale
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
            print(f"Warning: HOG extraction failed: {e}")
            hog_feat = np.zeros(1296)
        
        # 2. Color Histogram
        if len(img_resized.shape) == 3:
            hist = []
            for i in range(3):  # RGB channels
                h, _ = np.histogram(img_resized[:,:,i], bins=32, range=(0, 1))
                hist.append(h)
            color_hist = np.concatenate(hist)
        else:
            color_hist, _ = np.histogram(img_resized, bins=32, range=(0, 1))
        
        # 3. Edge Features
        edges = sobel(gray)
        edge_density = np.sum(edges > edges.mean()) / edges.size
        
        # 4. Statistical Features
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Combine all features (MUST match training order)
        features = np.concatenate([
            hog_feat, 
            color_hist, 
            [edge_density, mean_val, std_val]
        ])
        
    else:
        # === DEEP FEATURES ===
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for deep features")
        
        # Ensure 3 channels (RGB)
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        # Scale to 0-255 (matching training preprocessing)
        img_resized = (img_resized * 255).astype(np.uint8)
        img_batch = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_batch.astype(float))
        
        # Load model and extract features (cached to avoid reloading)
        global base_model_cache
        if base_model_cache is None:
            base_model_cache = MobileNetV2(weights='imagenet', include_top=False,
                                          input_shape=(224, 224, 3))
        features = base_model_cache.predict(img_preprocessed, verbose=0)
        features = features.flatten()
    
    return features.reshape(1, -1)

def predict_image(image_path, model, scaler, pca, class_names, feature_type):
    """Predict the class of a single image"""
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Extract features (MUST match training)
    features = extract_features_from_image(img_array, feature_type)
    
    # Scale features (using training scaler)
    features_scaled = scaler.transform(features)
    
    # Apply PCA if it was used in training
    if pca is not None:
        features_final = pca.transform(features_scaled)
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
        
        # Add percentage labels
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

def batch_predict(image_folder, model, scaler, pca, class_names, feature_type):
    """Predict on multiple images"""
    print("\n" + "="*70)
    print("BATCH PREDICTION")
    print("="*70)
    
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    results = []
    for img_path in image_files:
        try:
            predicted_class, probabilities, _ = predict_image(
                img_path, model, scaler, pca, class_names, feature_type
            )
            
            confidence = None
            if probabilities is not None:
                confidence = probabilities[class_names.index(predicted_class)] * 100
            
            results.append({
                'image': img_path.name,
                'predicted_class': predicted_class,
                'confidence': confidence
            })
            
            if confidence:
                print(f"{img_path.name}: {predicted_class} ({confidence:.1f}% confidence)")
            else:
                print(f"{img_path.name}: {predicted_class}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    return results

def test_on_validation_set(model, scaler, pca, class_names, feature_type):
    """Test model on validation set and calculate accuracy"""
    print("\n" + "="*70)
    print("TESTING ON VALIDATION SET")
    print("="*70)

    # Load validation data
    images, labels = load_images_and_labels("validation")
    print(f"Loaded {len(images)} validation images")

    correct = 0
    total = 0
    for img, true_label in zip(images, labels):
        # Extract features
        features = extract_features_from_image(img, feature_type)
        features_scaled = scaler.transform(features)
        features_final = pca.transform(features_scaled) if pca is not None else features_scaled

        pred = model.predict(features_final)[0]

        if pred == true_label:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"{'='*70}")

if __name__ == "__main__":
    print("="*70)
    print("ROCK-PAPER-SCISSORS PREDICTION")
    print("="*70)
    
    # Find available models
    models_dir = Path('models')
    available_models = []
    
    if (models_dir / 'best_model_handcrafted.pkl').exists():
        available_models.append(('handcrafted', 'models/best_model_handcrafted.pkl'))
    if (models_dir / 'best_model_deep_mobilenet.pkl').exists():
        available_models.append(('deep', 'models/best_model_deep_mobilenet.pkl'))
    
    if not available_models:
        print("\nError: No trained models found!")
        print("Please run train.py first.")
        exit()
    
    # Select model
    if len(available_models) == 1:
        model_path = available_models[0][1]
        print(f"\nUsing model: {model_path}")
    else:
        print("\nAvailable models:")
        for i, (name, path) in enumerate(available_models, 1):
            print(f"{i}. {name}")
        choice = input("Select model (1/2): ").strip()
        model_path = available_models[int(choice)-1][1]
    
    # Load model
    model, scaler, pca, class_names, feature_type = load_model(model_path)
    
    # Prediction options
    print("\n" + "="*70)
    print("PREDICTION OPTIONS")
    print("="*70)
    print("1. Predict single image")
    print("2. Predict batch of images")
    print("3. Test on validation set")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        # Single image prediction
        image_path = input("Enter image path: ").strip()
        
        try:
            predicted_class, probabilities, img = predict_image(
                image_path, model, scaler, pca, class_names, feature_type
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
        # Batch prediction
        folder_path = input("Enter folder path: ").strip()
        try:
            results = batch_predict(folder_path, model, scaler, pca, class_names, feature_type)
        except Exception as e:
            print(f"Error: {e}")
        
    elif choice == '3':
        # Test on validation set
        test_on_validation_set(model, scaler, pca, class_names, feature_type)
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETED!")
    print("="*70)