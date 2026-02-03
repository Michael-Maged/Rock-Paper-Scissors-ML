import os
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Import MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Error: MediaPipe not available - {e}")
    MEDIAPIPE_AVAILABLE = False
    hands = None

# Paths
MODELS_DIR = "models"

def extract_hand_landmarks(image):
    """Extract 21 hand landmarks using MediaPipe"""
    # Convert to RGB if needed
    if len(image.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image
    
    # Process the image
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks), hand_landmarks
    
    return None, None

def calculate_relative_distances(hand_landmarks):
    """Calculate relative distances between key landmarks"""
    if hand_landmarks is None:
        return np.zeros(15)
    
    landmarks = hand_landmarks.landmark
    
    def get_coords(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    
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
    
    # Distance between fingertips
    features.append(euclidean_distance(thumb_tip, index_tip))
    features.append(euclidean_distance(index_tip, middle_tip))
    features.append(euclidean_distance(middle_tip, ring_tip))
    features.append(euclidean_distance(ring_tip, pinky_tip))
    features.append(euclidean_distance(thumb_tip, pinky_tip))
    
    # Distance from fingertips to palm center
    palm_center = (get_coords(0) + get_coords(5) + get_coords(9) + 
                   get_coords(13) + get_coords(17)) / 5
    
    features.append(euclidean_distance(thumb_tip, palm_center))
    features.append(euclidean_distance(index_tip, palm_center))
    features.append(euclidean_distance(middle_tip, palm_center))
    features.append(euclidean_distance(ring_tip, palm_center))
    features.append(euclidean_distance(pinky_tip, palm_center))
    
    return np.array(features)

def calculate_finger_angles(hand_landmarks):
    """Calculate angles of fingers"""
    if hand_landmarks is None:
        return np.zeros(5)
    
    landmarks = hand_landmarks.landmark
    
    def get_coords(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
    
    def calculate_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle
    
    angles = []
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
    """Extract all hand-based features from an image"""
    landmarks_flat, hand_landmarks = extract_hand_landmarks(image)
    
    if landmarks_flat is None:
        return None, None
    
    distances = calculate_relative_distances(hand_landmarks)
    angles = calculate_finger_angles(hand_landmarks)
    
    all_features = np.concatenate([landmarks_flat, distances, angles])
    
    return all_features, hand_landmarks

def load_model():
    """Load the trained model"""
    print("\n" + "="*70)
    print("  🤖 LOADING MODEL...")
    print("="*70)
    
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please run train.py first.")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = model_data['class_names']
    
    print(f"✅ Model loaded: {type(model).__name__}")
    if 'accuracy' in model_data:
        print(f"✅ Training accuracy: {model_data['accuracy']:.2f}%")
    print(f"✅ Classes: {', '.join(class_names)}")
    print("="*70)
    
    return model, scaler, class_names

def predict_image(image_path, model, scaler, class_names):
    """Predict the class of a single image"""
    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)
    
    # Extract features
    features, hand_landmarks = extract_all_features(img_array)
    
    if features is None:
        return None, None, None, img_array
    
    features = features.reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    predicted_class = class_names[prediction]
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        probabilities = None
    
    return predicted_class, probabilities, hand_landmarks, img_array

def visualize_prediction(image, predicted_class, probabilities, hand_landmarks, class_names):
    """Visualize prediction result with hand landmarks"""
    fig = plt.figure(figsize=(15, 5))
    
    # Image with hand landmarks
    ax1 = plt.subplot(1, 3, 1)
    if hand_landmarks is not None:
        # Draw landmarks on image
        image_with_landmarks = image.copy()
        h, w = image.shape[:2]
        
        # Draw connections
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = hand_landmarks.landmark[start_idx]
            end_point = hand_landmarks.landmark[end_idx]
            
            start_x, start_y = int(start_point.x * w), int(start_point.y * h)
            end_x, end_y = int(end_point.x * w), int(end_point.y * h)
            
            cv2.line(image_with_landmarks, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image_with_landmarks, (x, y), 5, (255, 0, 0), -1)
        
        ax1.imshow(image_with_landmarks)
    else:
        ax1.imshow(image)
    
    ax1.set_title('Hand Landmarks Detected', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Original image
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(image)
    ax2.set_title(f'Predicted: {predicted_class.upper()}',
                  fontsize=16, fontweight='bold', color='green')
    ax2.axis('off')
    
    # Probabilities
    ax3 = plt.subplot(1, 3, 3)
    if probabilities is not None:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax3.barh(class_names, probabilities * 100, color=colors)
        ax3.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Class Probabilities', fontsize=14, fontweight='bold')
        ax3.set_xlim([0, 100])
        
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            ax3.text(width + 2, bar.get_y() + bar.get_height()/2,
                    f'{prob*100:.1f}%', va='center', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, f'Predicted:\n{predicted_class.upper()}',
                ha='center', va='center', fontsize=18, fontweight='bold')
        ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def test_on_folder(folder_path, model, scaler, class_names):
    """Test on all images in a folder"""
    print("\n" + "="*70)
    print(f"  📂 TESTING FOLDER: {os.path.basename(folder_path)}")
    print("="*70)
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = sorted([f for f in os.listdir(folder_path)
                         if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print("❌ No images found in folder!")
        return
    
    print(f"📊 Found {len(image_files)} images\n")
    
    correct = 0
    total = 0
    no_hand_detected = 0
    predictions_by_class = {'rock': [], 'paper': [], 'scissors': []}
    
    # Progress tracking
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            predicted_class, probabilities, hand_landmarks, _ = predict_image(
                img_path, model, scaler, class_names
            )
            
            if predicted_class is None:
                print(f"[{idx}/{len(image_files)}] ⚠️  {img_file}: No hand detected")
                no_hand_detected += 1
                continue
            
            # Extract true class from filename
            true_class = None
            for cls in class_names:
                if cls in img_file.lower():
                    true_class = cls
                    break
            
            # Build result string
            confidence = probabilities[class_names.index(predicted_class)] * 100
            
            # Emoji for each class
            emojis = {'rock': '✊', 'paper': '✋', 'scissors': '✌️'}
            pred_emoji = emojis.get(predicted_class, '👋')
            
            is_correct = (true_class == predicted_class) if true_class else None
            
            if true_class:
                if is_correct:
                    status = "✅"
                    correct += 1
                else:
                    status = "❌"
                    true_emoji = emojis.get(true_class, '👋')
                total += 1
                
                result_str = f"[{idx}/{len(image_files)}] {status} {img_file[:30]:30} → {pred_emoji} {predicted_class.upper():8} ({confidence:5.1f}%) [True: {true_class}]"
            else:
                result_str = f"[{idx}/{len(image_files)}] {pred_emoji} {img_file[:30]:30} → {predicted_class.upper():8} ({confidence:5.1f}%)"
            
            print(result_str)
            predictions_by_class[predicted_class].append(img_file)
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ❌ {img_file}: Error - {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("  📊 RESULTS SUMMARY")
    print("="*70)
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"✅ Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
    
    if no_hand_detected > 0:
        print(f"⚠️  No hand detected: {no_hand_detected} images")
    
    # Predictions breakdown
    print("\n📈 Predictions by class:")
    for cls in class_names:
        count = len(predictions_by_class[cls])
        emojis = {'rock': '✊', 'paper': '✋', 'scissors': '✌️'}
        emoji = emojis.get(cls, '👋')
        print(f"   {emoji} {cls.capitalize():8}: {count} images")
    
    print("="*70)

def main():
    """Main function"""
    if not MEDIAPIPE_AVAILABLE:
        print("\n❌ ERROR: MediaPipe is required for prediction!")
        print("   Install it with: pip install mediapipe")
        return
    
    print("\n" + "="*70)
    print("  ✊✋✌️  ROCK-PAPER-SCISSORS CLASSIFIER")
    print("  🤖 Hand Landmark Detection with Machine Learning")
    print("="*70)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    model, scaler, class_names = model_data
    
    # Prediction options
    print("\n" + "="*70)
    print("  🎯 PREDICTION OPTIONS")
    print("="*70)
    print("  1️⃣  Predict single image (with visualization)")
    print("  2️⃣  Test on folder (batch processing)")
    print("="*70)
    
    choice = input("\n👉 Select option (1/2): ").strip()
    
    if choice == '1':
        # Single image prediction
        image_path = input("📁 Enter image path: ").strip().strip('"')
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image not found at {image_path}")
            return
        
        try:
            print("\n🔍 Analyzing image...")
            predicted_class, probabilities, hand_landmarks, img = predict_image(
                image_path, model, scaler, class_names
            )
            
            if predicted_class is None:
                print("\n" + "="*70)
                print("❌ ERROR: No hand detected in the image!")
                print("   Make sure the image contains a clear hand gesture.")
                print("="*70)
                return
            
            # Emoji mapping
            emojis = {'rock': '✊', 'paper': '✋', 'scissors': '✌️'}
            emoji = emojis.get(predicted_class, '👋')
            
            print("\n" + "="*70)
            print("  🎯 PREDICTION RESULT")
            print("="*70)
            print(f"  {emoji} Predicted: {predicted_class.upper()}")
            
            if probabilities is not None:
                print("\n  📊 Confidence Scores:")
                for cls, prob in zip(class_names, probabilities):
                    cls_emoji = emojis.get(cls, '👋')
                    bar_length = int(prob * 30)
                    bar = '█' * bar_length + '░' * (30 - bar_length)
                    print(f"     {cls_emoji} {cls.capitalize():8} {bar} {prob*100:5.1f}%")
            print("="*70)
            
            print("\n📊 Opening visualization...")
            # Visualize
            visualize_prediction(img, predicted_class, probabilities, hand_landmarks, class_names)
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == '2':
        # Test on folder
        folder_path = input("📁 Enter folder path: ").strip().strip('"')
        
        if not os.path.exists(folder_path):
            print(f"❌ Error: Folder not found at {folder_path}")
            return
        
        test_on_folder(folder_path, model, scaler, class_names)
    
    else:
        print("❌ Invalid option! Please select 1 or 2.")
    
    print("\n" + "="*70)
    print("  ✅ PREDICTION COMPLETED!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()