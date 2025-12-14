import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import sobel
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
    # Resize
    img_resized = resize(img_array, target_size, anti_aliasing=True)
    
    # Handle RGBA
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]
    
    if feature_type == 'handcrafted':
        # Extract handcrafted features
        if len(img_resized.shape) == 3:
            gray = np.mean(img_resized, axis=2)
        else:
            gray = img_resized
        
        # HOG
        hog_feat = hog(gray, pixels_per_cell=(16, 16), 
                      cells_per_block=(2, 2), feature_vector=True)
        
        # Color histogram
        if len(img_resized.shape) == 3:
            hist = []
            for i in range(3):
                h, _ = np.histogram(img_resized[:,:,i], bins=32, range=(0, 1))
                hist.append(h)
            color_hist = np.concatenate(hist)
        else:
            color_hist, _ = np.histogram(img_resized, bins=32, range=(0, 1))
        
        # Edge features
        edges = sobel(gray)
        edge_density = np.sum(edges > edges.mean()) / edges.size
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        features = np.concatenate([hog_feat, color_hist, [edge_density, mean_val, std_val]])
        
    else:  # deep features
        # For deep features, you'd need to use the CNN model
        # This is simplified - in practice, load the CNN and extract features
        
        # Ensure 3 channels
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        # Scale to 0-255
        img_resized = (img_resized * 255).astype(np.uint8)
        img_batch = np.expand_dims(img_resized, axis=0)
        img_preprocessed = preprocess_input(img_batch.astype(float))
        
        # Load model and extract features
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(224, 224, 3))
        features = base_model.predict(img_preprocessed, verbose=0)
        features = features.flatten()
    
    return features.reshape(1, -1)

def predict_image(image_path, model, scaler, pca, class_names, feature_type):
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Extract features
    features = extract_features_from_image(img_array, feature_type)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Apply PCA if available
    if pca is not None:
        features_final = pca.transform(features_scaled)
    else:
        features_final = features_scaled
    
    # Predict
    prediction = model.predict(features_final)[0]
    predicted_class = class_names[prediction]
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_final)[0]
    else:
        probabilities = None
    
    return predicted_class, probabilities, img

def visualize_prediction(image, predicted_class, probabilities, class_names):
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
    print("\n" + "="*70)
    print("BATCH PREDICTION")
    print("="*70)
    
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.png')) + list(image_folder.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    results = []
    for img_path in image_files:
        predicted_class, probabilities, _ = predict_image(
            img_path, model, scaler, pca, class_names, feature_type
        )
        results.append({
            'image': img_path.name,
            'predicted_class': predicted_class,
            'confidence': probabilities[class_names.index(predicted_class)] * 100 if probabilities is not None else None
        })
        print(f"{img_path.name}: {predicted_class} ({results[-1]['confidence']:.1f}% confidence)" 
              if results[-1]['confidence'] else f"{img_path.name}: {predicted_class}")
    
    return results

if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("ROCK-PAPER-SCISSORS PREDICTION")
    print("="*70)
    
    # Load model
    model_path = 'models/best_model_handcrafted.pkl'
    
    # Check if deep model exists
    if Path('models/best_model_deep_mobilenet.pkl').exists():
        print("\nAvailable models:")
        print("1. Handcrafted features")
        print("2. Deep features (MobileNetV2)")
        choice = input("Select model (1/2): ").strip()
        if choice == '2':
            model_path = 'models/best_model_deep_mobilenet.pkl'
    
    model, scaler, pca, class_names, feature_type = load_model(model_path)
    
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
        predicted_class, probabilities, img = predict_image(
            image_path, model, scaler, pca, class_names, feature_type
        )
        
        print(f"\nPredicted class: {predicted_class.upper()}")
        if probabilities is not None:
            print("\nProbabilities:")
            for cls, prob in zip(class_names, probabilities):
                print(f"  {cls}: {prob*100:.2f}%")
        
        # Visualize
        visualize_prediction(img, predicted_class, probabilities, class_names)
        
    elif choice == '2':
        # Batch prediction
        folder_path = input("Enter folder path: ").strip()
        results = batch_predict(folder_path, model, scaler, pca, class_names, feature_type)
        
    elif choice == '3':
        # Test on validation set
        print("\nTesting on validation set...")
        val_path = Path(__file__).parent.parent / "data" / "validation"
        results = batch_predict(val_path, model, scaler, pca, class_names, feature_type)
        
        # Calculate accuracy
        correct = 0
        total = 0
        for result in results:
            img_name = result['image']
            predicted = result['predicted_class']
            # Extract actual class from filename (e.g., "rock01.png" -> "rock")
            actual = None
            for cls in class_names:
                if img_name.startswith(cls):
                    actual = cls
                    break
            if actual and actual == predicted:
                correct += 1
            total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\nValidation Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETED!")
    print("="*70)