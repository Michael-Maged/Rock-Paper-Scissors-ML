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
from sklearn.feature_selection import SelectKBest, f_classif
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
    print("="*70)
    print("LOADING FEATURES")
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
    
    return X_train, X_test, y_train, y_test

def preprocess_features(X_train, X_test, y_train, use_feature_selection=True, k_features=50):
    print(f"\n" + "="*70)
    print(f"FEATURE PREPROCESSING")
    print(f"="*70)
    
    # Standardize features
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if use_feature_selection:
        print(f"Selecting top {k_features} features using ANOVA F-test...")
        selector = SelectKBest(f_classif, k=min(k_features, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        print(f"Reduced from {X_train_scaled.shape[1]} to {X_train_selected.shape[1]} features")
        
        return X_train_selected, X_test_selected, scaler, selector
    else:
        return X_train_scaled, X_test_scaled, scaler, None

def get_classifiers():
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,  # Shallower to avoid overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # Fewer trees
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,  # Very shallow
            learning_rate=0.3,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,  # More neighbors for stability
            weights='distance',
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=0.1,  # Lower C to prevent overfitting
            gamma='scale',
            random_state=42,
            probability=True,
            class_weight='balanced'
        ),
        'ANN': MLPClassifier(
            hidden_layer_sizes=(32,),  # Single small layer
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            alpha=0.01  # Regularization
        )
    }
    return classifiers

def plot_confusion_matrix(cm, class_names, classifier_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {classifier_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'cm_{classifier_name.lower().replace(" ", "_")}.png'
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate(X_train, y_train, X_test, y_test):
    classifiers = get_classifiers()
    
    print(f"\n" + "="*70)
    print(f"TRAINING AND EVALUATION")
    print(f"="*70)
    
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
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
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
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))
    
    return results, trained_models

def save_results(results):
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy (%)', ascending=False)
    
    print(f"\n" + "="*70)
    print(f"RESULTS SUMMARY")
    print(f"="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Save to CSV
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)
    print(f"\nResults saved to: {RESULTS_DIR}/results.csv")
    
    # Plot 1: Accuracy Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    
    plt.bar(x - width/2, results_df['Train Accuracy (%)'], width,
            label='Train', color='#2E86AB', alpha=0.8)
    plt.bar(x + width/2, results_df['Test Accuracy (%)'], width,
            label='Test', color='#A23B72', alpha=0.8)
    
    plt.xlabel('Classifier', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Classifier'], rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
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
        plt.text(time_val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{time_val:.3f}s', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_time.png'), dpi=300)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/training_time.png")
    
    # Plot 3: Metrics Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
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
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved: {RESULTS_DIR}/metrics_comparison.png")
    
    return results_df

def save_best_model(results_df, trained_models, scaler, selector):
    best_model_name = results_df.iloc[0]['Classifier']
    best_model = trained_models[best_model_name]
    best_accuracy = results_df.iloc[0]['Test Accuracy (%)']
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Test Accuracy: {best_accuracy:.2f}%")
    print('='*70)
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'selector': selector,
        'class_names': CLASS_NAMES
    }
    
    filename = os.path.join(MODELS_DIR, 'best_model.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nBest model saved to: {filename}")

def main():
    # Check if features exist
    if not os.path.exists(FEATURES_DIR):
        print(f"Error: Features directory '{FEATURES_DIR}' not found!")
        print("Please run feature_extraction.py first.")
        return
    
    # Load features
    X_train, X_test, y_train, y_test = load_features()
    
    # Preprocess features (NOW PASSING y_train)
    X_train_proc, X_test_proc, scaler, selector = preprocess_features(
        X_train, X_test, y_train, use_feature_selection=False
    )
    
    # Train and evaluate
    results, trained_models = train_and_evaluate(
        X_train_proc, y_train, X_test_proc, y_test
    )
    
    # Save results
    results_df = save_results(results)
    
    # Save best model
    save_best_model(results_df, trained_models, scaler, selector)
    
    print(f"\n" + "="*70)
    print(f"TRAINING COMPLETED!")
    print(f"="*70)
    print(f"Generated files:")
    print(f"- Confusion matrices: {RESULTS_DIR}/cm_*.png")
    print(f"- Comparison plots: {RESULTS_DIR}/")
    print(f"- Results: {RESULTS_DIR}/results.csv")
    print(f"- Best model: {MODELS_DIR}/best_model.pkl")
    print(f"\nNext step: Run predict.py to test on new images")
    print(f"="*70)

if __name__ == "__main__":
    main()