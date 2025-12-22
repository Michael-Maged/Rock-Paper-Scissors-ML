"""
Simpler CNN approach - trains faster and more reliably
Good for quick testing and improvement
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"

def load_images_simple(split='training', target_size=(160, 160)):
    """Faster image loading for CNN with error handling"""
    images = []
    labels = []
    label_map = {"rock": 0, "paper": 1, "scissors": 2}
    skipped = 0
    
    if split == "validation":
        path = DATA_DIR / "validation"
        if path.exists():
            for gesture, label in label_map.items():
                all_files = [f for f in path.glob("*.png") if gesture in f.name]
                for img_path in all_files:
                    try:
                        from PIL import Image
                        img = Image.open(str(img_path)).convert('RGB')
                        img = img.resize(target_size)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except Exception :
                        skipped += 1
    else:
        for gesture, label in label_map.items():
            path = DATA_DIR / split / gesture
            if path.exists():
                all_files = list(path.glob("*.png"))
                for img_path in all_files:
                    try:
                        from PIL import Image
                        img = Image.open(str(img_path)).convert('RGB')
                        img = img.resize(target_size)
                        img_array = np.array(img) / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except Exception :
                        skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} corrupted images")
    
    return np.array(images), np.array(labels)

def build_cnn_model(input_shape=(160, 160, 3)):
    """Build simple CNN model"""
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(3, activation='softmax')
    ])
    
    return model

def train_simple_cnn():
    """Train simple CNN with data augmentation"""
    print("="*70)
    print("SIMPLE CNN TRAINING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train = load_images_simple('training')
    X_val, y_val = load_images_simple('validation')
    X_test, y_test = load_images_simple('testing')
    
    print(f"Training: {X_train.shape[0]} images")
    print(f"Validation: {X_val.shape[0]} images")
    print(f"Testing: {X_test.shape[0]} images")
    
    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    model = build_cnn_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = model.fit(
        aug.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=25,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['rock', 'paper', 'scissors'],
                              zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['rock', 'paper', 'scissors'],
                yticklabels=['rock', 'paper', 'scissors'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Simple CNN')
    plt.tight_layout()
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/cm_simple_cnn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrix saved")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/best_model_simple_cnn.keras')
    print("Model saved to: models/best_model_simple_cnn.keras")
    
    # Plot history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/training_history_simple_cnn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved")

if __name__ == "__main__":
    train_simple_cnn()
