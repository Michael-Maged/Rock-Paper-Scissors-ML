import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2
from feature_extraction import extract_all_features

# Paths
MODELS_DIR = "models"

def load_model():
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
            
            # Extract true class from filename (r=rock, p=paper, s=scissors)
            true_class = None
            first_char = img_file.lower()[0]
            if first_char == 'r':
                true_class = 'rock'
            elif first_char == 'p':
                true_class = 'paper'
            elif first_char == 's':
                true_class = 'scissors'
            
            if probabilities is not None:
                confidence = probabilities[class_names.index(predicted_class)] * 100
                result_str = f"{img_file}: {predicted_class} ({confidence:.1f}%)"
            else:
                result_str = f"{img_file}: {predicted_class}"
            
            if true_class:
                result_str += f" [Actual: {true_class}]"
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
        print(f"PREDICTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total images: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Wrong predictions: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"{'='*70}")
    else:
        print(f"\nNo images with recognizable naming pattern found.")
        print(f"Expected: filenames starting with 'r' (rock), 'p' (paper), or 's' (scissors)")

def main():
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