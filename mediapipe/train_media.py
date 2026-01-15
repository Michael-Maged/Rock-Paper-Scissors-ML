import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import xgboost as xgb

# Paths
FEATURES_DIR = "features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ['rock', 'paper', 'scissors']

def load_features():
    """Load extracted features"""
    print("="*70)
    print("LOADING HAND LANDMARK FEATURES")
    print("="*70)
    
    with open(os.path.join(FEATURES_DIR, 'train_features.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    
    with open(os.path.join(FEATURES_DIR, 'test_features.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    print("  - Hand landmarks (x,y,z): 63 features")
    print("  - Relative distances: 15 features")
    print("  - Finger angles: 5 features")
    
    # Check class distribution
    print("\nClass distribution:")
    for i, class_name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {class_name}: {train_count} train, {test_count} test")
    
    return X_train, X_test, y_train, y_test

def preprocess_features(X_train, X_test):
    """
    Preprocess features: scaling only
    Note: We keep all hand landmark features as they're all meaningful
    """
    print("\n" + "="*70)
    print("FEATURE PREPROCESSING")
    print("="*70)
    
    # Standardize features (important for distance-based classifiers)
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled to zero mean and unit variance")
    print("Using all {X_train_scaled.shape[1]} hand landmark features")
    
    return X_train_scaled, X_test_scaled, scaler

def get_classifiers():
    """Define all classifiers optimized for hand pose recognition"""
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='euclidean',
            n_jobs=-1
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            random_state=42,
            probability=True,
            class_weight='balanced'
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            C=1.0,
            random_state=42,
            probability=True,
            class_weight='balanced'
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    return classifiers

def plot_confusion_matrix(cm, class_names, classifier_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {classifier_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'cm_{classifier_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train all classifiers and evaluate"""
    classifiers = get_classifiers()
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION")
    print("="*70)
    
    results = []
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print('='*70)
        
        # Train
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Store results
        results.append({
            'Classifier': name,
            'Training Time (s)': round(training_time, 3),
            'Train Accuracy (%)': round(train_accuracy * 100, 2),
            'Test Accuracy (%)': round(test_accuracy * 100, 2),
            'Precision': round(test_precision, 4),
            'Recall': round(test_recall, 4),
            'F1-Score': round(test_f1, 4)
        })
        
        trained_models[name] = clf
        
        # Print metrics
        print(f"Training Time: {training_time:.3f}s")
        print(f"Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm, CLASS_NAMES, name)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, zero_division=0))
        
        # Print per-class accuracy
        print("\nPer-class Accuracy:")
        for i, class_name in enumerate(CLASS_NAMES):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_test_pred[class_mask])
                print(f"  {class_name}: {class_acc*100:.2f}%")
    
    return results, trained_models

def save_results(results):
    """Save results to CSV and generate comparison plots"""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy (%)', ascending=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    print(f"\nResults saved to: {RESULTS_DIR}/results.csv")
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Train Accuracy (%)'], width,
            label='Train', color='#2E86AB', alpha=0.8)
    plt.bar(x + width/2, results_df['Test Accuracy (%)'], width,
            label='Test', color='#A23B72', alpha=0.8)
    
    plt.xlabel('Classifier', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison - Hand Landmark Models', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Classifier'], rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 105)
    
    # Add value labels on bars
    for i, (train_acc, test_acc) in enumerate(zip(results_df['Train Accuracy (%)'], 
                                                    results_df['Test Accuracy (%)'])):
        plt.text(i - width/2, train_acc + 1, f'{train_acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, test_acc + 1, f'{test_acc:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/accuracy_comparison.png")
    
    # Plot 2: Training Time Comparison
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    bars = plt.barh(results_df['Classifier'], results_df['Training Time (s)'], color=colors)
    plt.xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Classifier', fontsize=12, fontweight='bold')
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for bar, time_val in zip(bars, results_df['Training Time (s)']):
        plt.text(time_val + max(results_df['Training Time (s)'])*0.02, 
                bar.get_y() + bar.get_height()/2,
                f'{time_val:.3f}s', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_time.png'), dpi=300)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/training_time.png")
    
    # Plot 3: Metrics Comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['Precision'], width, label='Precision',
           color='#06D6A0', alpha=0.8)
    ax.bar(x, results_df['Recall'], width, label='Recall',
           color='#FFD166', alpha=0.8)
    ax.bar(x + width, results_df['F1-Score'], width, label='F1-Score',
           color='#EF476F', alpha=0.8)
    
    ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1-Score Comparison',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Classifier'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/metrics_comparison.png")
    
    return results_df

def save_best_model(results_df, trained_models, scaler):
    """Save the best performing model"""
    best_model_name = results_df.iloc[0]['Classifier']
    best_model = trained_models[best_model_name]
    best_accuracy = results_df.iloc[0]['Test Accuracy (%)']
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.2f}%")
    print('='*70)
    
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'class_names': CLASS_NAMES,
        'feature_type': 'hand_landmarks',
        'accuracy': best_accuracy
    }
    
    filename = os.path.join(MODELS_DIR, 'best_model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nBest model saved to: {filename}")
    
    # Also save all models
    all_models_data = {
        'models': trained_models,
        'scaler': scaler,
        'class_names': CLASS_NAMES,
        'results': results_df
    }
    
    filename_all = os.path.join(MODELS_DIR, 'all_models.pkl')
    with open(filename_all, 'wb') as f:
        pickle.dump(all_models_data, f)
    
    print(f"All models saved to: {filename_all}")

def main():
    """Main function"""
    # Check if features exist
    if not os.path.exists(FEATURES_DIR):
        print(f"Error: Features directory '{FEATURES_DIR}' not found!")
        print("Please run feature_extraction_media.py first.")
        return
    
    # Load features
    X_train, X_test, y_train, y_test = load_features()
    
    # Preprocess features (scaling only, no feature selection for hand landmarks)
    X_train_proc, X_test_proc, scaler = preprocess_features(X_train, X_test)
    
    # Train and evaluate
    results, trained_models = train_and_evaluate(
        X_train_proc, y_train, X_test_proc, y_test
    )
    
    # Save results
    results_df = save_results(results)
    
    # Save best model
    save_best_model(results_df, trained_models, scaler)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print("\nGenerated files:")
    print("  Models:")
    print(f"    - {MODELS_DIR}/best_model.pkl")
    print(f"    - {MODELS_DIR}/all_models.pkl")
    print("  Results:")
    print(f"    - {RESULTS_DIR}/results.csv")
    print(f"    - {RESULTS_DIR}/accuracy_comparison.png")
    print(f"    - {RESULTS_DIR}/training_time.png")
    print(f"    - {RESULTS_DIR}/metrics_comparison.png")
    print("  Confusion Matrices:")
    print(f"    - {RESULTS_DIR}/cm_*.png (one per classifier)")
    
    print("\n" + "="*70)
    print("BEST MODEL DETAILS")
    print("="*70)
    print(f"Model: {results_df.iloc[0]['Classifier']}")
    print(f"Test Accuracy: {results_df.iloc[0]['Test Accuracy (%)']}%")
    print(f"F1-Score: {results_df.iloc[0]['F1-Score']}")
    print("\nNext step: Run predict.py to test on new images")
    print("="*70)

if __name__ == "__main__":
    main()