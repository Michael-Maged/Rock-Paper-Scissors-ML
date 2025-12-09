from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from skimage.feature import hog
from skimage.filters import sobel
from skimage.transform import resize
import argparse

MODEL_DIR = Path(__file__).parent.parent / "models"
LABEL_MAP = {0: "rock", 1: "paper", 2: "scissors"}

def extract_features_from_image(img_path, target_size=(224, 224)):
    """Extract handcrafted features from a single image"""
    
    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Resize and normalize
    img_resized = resize(img_array, target_size, anti_aliasing=True)
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
    
    return combined.reshape(1, -1)

def load_model_and_scaler():
    """Load the trained model and scaler"""
    
    model_path = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def predict_single_image(img_path, model=None, scaler=None, verbose=True):
    """Predict the gesture in a single image"""
    
    # Load model and scaler if not provided
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Extract features
    if verbose:
        print(f"Processing image: {img_path}")
    features = extract_features_from_image(img_path)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        probabilities = None
    
    gesture = LABEL_MAP[prediction]
    
    if verbose:
        print(f"\nPrediction: {gesture.upper()}")
        if probabilities is not None:
            print("\nConfidence scores:")
            for label_idx, label_name in LABEL_MAP.items():
                print(f"  {label_name}: {probabilities[label_idx]:.2%}")
    
    return gesture, probabilities

def predict_batch(image_dir, model=None, scaler=None):
    """Predict gestures for all images in a directory"""
    
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Load model and scaler once
    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()
    
    # Get all image files
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("="*60)
    
    results = []
    
    for img_path in image_files:
        gesture, probabilities = predict_single_image(img_path, model, scaler, verbose=False)
        results.append({
            "filename": img_path.name,
            "prediction": gesture,
            "probabilities": probabilities
        })
        
        print(f"{img_path.name:30s} -> {gesture.upper()}")
    
    print("="*60)
    
    # Summary
    predictions = [r["prediction"] for r in results]
    print("\nSummary:")
    for gesture in ["rock", "paper", "scissors"]:
        count = predictions.count(gesture)
        print(f"  {gesture}: {count} ({count/len(predictions)*100:.1f}%)")
    
    return results

def interactive_mode():
    """Interactive prediction mode"""
    
    print("="*60)
    print("ROCK PAPER SCISSORS - INTERACTIVE PREDICTION")
    print("="*60)
    
    # Load model and scaler once
    model, scaler = load_model_and_scaler()
    print("\nModel loaded successfully!")
    
    while True:
        print("\n" + "-"*60)
        img_path = input("Enter image path (or 'quit' to exit): ").strip()
        
        if img_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not Path(img_path).exists():
            print(f"Error: File not found - {img_path}")
            continue
        
        try:
            predict_single_image(img_path, model, scaler, verbose=True)
        except Exception as e:
            print(f"Error processing image: {e}")

def main():
    parser = argparse.ArgumentParser(description="Rock Paper Scissors Gesture Prediction")
    parser.add_argument("--image", "-i", type=str, help="Path to a single image")
    parser.add_argument("--batch", "-b", type=str, help="Path to directory with images")
    parser.add_argument("--interactive", "-int", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.image:
        predict_single_image(args.image)
    elif args.batch:
        predict_batch(args.batch)
    elif args.interactive:
        interactive_mode()
    else:
        print("Please specify --image, --batch, or --interactive mode")
        print("Examples:")
        print("  python predict.py --image path/to/image.png")
        print("  python predict.py --batch path/to/images/")
        print("  python predict.py --interactive")

if __name__ == "__main__":
    main()