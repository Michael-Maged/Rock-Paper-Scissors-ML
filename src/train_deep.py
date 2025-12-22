"""
Deep Learning approach using fine-tuned MobileNetV2
This typically achieves 85-95% accuracy on rock-paper-scissors
"""
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"

def create_dataset(split='training', batch_size=32):
    """Load images directly and create tf.data.Dataset"""
    images = []
    labels = []
    label_map = {"rock": 0, "paper": 1, "scissors": 2}
    
    if split == "validation":
        path = DATA_DIR / "validation"
        if path.exists():
            for gesture, label in label_map.items():
                all_files = [f for f in path.glob("*.png") if gesture in f.name]
                for img_path in all_files:
                    try:
                        img = keras.preprocessing.image.load_img(
                            img_path, target_size=(224, 224)
                        )
                        img_array = keras.preprocessing.image.img_to_array(img)
                        # Normalize to [0, 1]
                        img_array = img_array / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Could not load {img_path}: {e}")
    else:
        for gesture, label in label_map.items():
            path = DATA_DIR / split / gesture
            if path.exists():
                all_files = list(path.glob("*.png"))
                for img_path in all_files:
                    try:
                        img = keras.preprocessing.image.load_img(
                            img_path, target_size=(224, 224)
                        )
                        img_array = keras.preprocessing.image.img_to_array(img)
                        # Normalize to [0, 1]
                        img_array = img_array / 255.0
                        images.append(img_array)
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Could not load {img_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    # Normalize using MobileNetV2 preprocessing
    from keras.applications.mobilenet_v2 import preprocess_input
    X = preprocess_input(X * 255.0)  # preprocess_input expects 0-255 range
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(X), X, y

def build_model():
    """Build fine-tuned MobileNetV2 model"""
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    model = keras.models.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    return model, base_model

def train_model():
    """Train the model with fine-tuning"""
    print("="*70)
    print("DEEP LEARNING TRAINING - MOBILENETV2")
    print("="*70)
    
    # Load datasets
    print("\nLoading training data...")
    train_dataset, train_size, X_train, y_train = create_dataset('training', batch_size=32)
    
    print(f"Training samples: {train_size}")
    
    print("Loading validation data...")
    val_dataset, val_size, X_val, y_val = create_dataset('validation', batch_size=32)
    print(f"Validation samples: {val_size}")
    
    print("Loading test data...")
    test_dataset, test_size, X_test, y_test = create_dataset('testing', batch_size=32)
    print(f"Test samples: {test_size}")
    
    # Build model
    print("\n" + "="*70)
    print("BUILDING MODEL")
    print("="*70)
    model, base_model = build_model()
    
    # Compile with frozen base model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train initial epochs
    print("\nPhase 1: Training with frozen base model...")
    history1 = model.fit(
        train_dataset,
        epochs=15,
        validation_data=val_dataset,
        verbose=1
    )
    
    # Unfreeze base model for fine-tuning
    print("\nPhase 2: Fine-tuning with unfrozen base model...")
    base_model.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Get predictions
    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Calculate metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"\nValidation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, 
                              target_names=['rock', 'paper', 'scissors']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['rock', 'paper', 'scissors'],
                yticklabels=['rock', 'paper', 'scissors'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Deep Learning (MobileNetV2)')
    plt.tight_layout()
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/cm_deep_learning.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/best_model_deep_learning.keras')
    print("\nModel saved to: models/best_model_deep_learning.keras")
    
    # Save training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['accuracy'], label='Phase 1 - Train')
    plt.plot(history1.history['val_accuracy'], label='Phase 1 - Val')
    plt.plot([len(history1.history['accuracy'])-1 + i for i in range(len(history2.history['accuracy']))],
             history2.history['accuracy'], label='Phase 2 - Train')
    plt.plot([len(history1.history['accuracy'])-1 + i for i in range(len(history2.history['val_accuracy']))],
             history2.history['val_accuracy'], label='Phase 2 - Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['loss'], label='Phase 1 - Train')
    plt.plot(history1.history['val_loss'], label='Phase 1 - Val')
    plt.plot([len(history1.history['loss'])-1 + i for i in range(len(history2.history['loss']))],
             history2.history['loss'], label='Phase 2 - Train')
    plt.plot([len(history1.history['loss'])-1 + i for i in range(len(history2.history['val_loss']))],
             history2.history['val_loss'], label='Phase 2 - Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/training_history_deep_learning.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training history saved to: results/plots/training_history_deep_learning.png")

if __name__ == "__main__":
    train_model()
