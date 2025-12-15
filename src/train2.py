import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import pickle
import time
import os
import warnings
warnings.filterwarnings('ignore')

def get_classifiers():
    """
    Define all classifiers to be trained
    """
    classifiers = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        'ANN': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    }
    
    return classifiers

def load_features(feature_type='handcrafted'):
    print("="*70)
    print(f"LOADING {feature_type.upper()} FEATURES")
    print("="*70)
    
    # Load training data
    with open(f'data/features/training_{feature_type}.pkl', 'rb') as f:
        train_data = pickle.load(f)
    X_train = train_data['features']
    y_train = train_data['labels']
    
    # Load validation data
    with open(f'data/features/validation_{feature_type}.pkl', 'rb') as f:
        val_data = pickle.load(f)
    X_val = val_data['features']
    y_val = val_data['labels']
    
    # Load testing data
    with open(f'data/features/testing_{feature_type}.pkl', 'rb') as f:
        test_data = pickle.load(f)
    X_test = test_data['features']
    y_test = test_data['labels']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_features(X_train, X_val, X_test, use_pca=True, n_components=None):
    print("\n" + "="*70)
    print("FEATURE PREPROCESSING")
    print("="*70)
    
    # Standardize features
    print("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    if use_pca:
        print("Applying PCA for dimensionality reduction...")
        if n_components is None:
            n_components = min(300, X_train_scaled.shape[1])
        
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"Reduced from {X_train_scaled.shape[1]} to {n_components} features")
        print(f"Explained variance: {explained_var:.2%}")
        
        # Plot PCA variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), linewidth=2, color='#2E86AB')
        plt.xlabel('Number of Components', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        plt.title('PCA - Explained Variance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/pca_variance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("PCA plot saved: results/plots/pca_variance.png")
        
        return X_train_pca, X_val_pca, X_test_pca, scaler, pca
    else:
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler, None

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, scaler, pca, feature_type):
    class_names = ['rock', 'paper', 'scissors']
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
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val, y_val_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Store results
        results.append({
            'Classifier': name,
            'Training Time (s)': round(training_time, 2),
            'Validation Accuracy (%)': round(val_accuracy * 100, 2),
            'Test Accuracy (%)': round(test_accuracy * 100, 2),
            'Precision': round(test_precision, 4),
            'Recall': round(test_recall, 4),
            'F1-Score': round(test_f1, 4)
        })
        
        # Store trained model
        trained_models[name] = clf
        
        # Print metrics
        print(f"Training Time: {training_time:.2f}s")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        
        # Print classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy (%)', ascending=False)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Save to CSV
    os.makedirs('results/metrics', exist_ok=True)
    results_df.to_csv(f'results/metrics/results_{feature_type}.csv', index=False)
    print(f"\nResults saved to: results/metrics/results_{feature_type}.csv")
    
    # Save best model
    best_model_name = results_df.iloc[0]['Classifier']
    best_model = trained_models[best_model_name]
    
    os.makedirs('models', exist_ok=True)
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'pca': pca,
        'class_names': class_names,
        'feature_type': feature_type
    }
    
    model_path = f'models/best_model_{feature_type}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nBest model ({best_model_name}) saved to: {model_path}")
    
    return results_df, trained_models

if __name__ == "__main__":
    print("="*70)
    print("ROCK-PAPER-SCISSORS CLASSIFICATION")
    print("="*70)
    
    # Check which features are available
    feature_dir = 'data/features'
    available_features = []
    
    if os.path.exists(f'{feature_dir}/training_handcrafted.pkl'):
        available_features.append('handcrafted')
    if os.path.exists(f'{feature_dir}/training_deep_mobilenet.pkl'):
        available_features.append('deep_mobilenet')
    
    if not available_features:
        print("\nError: No feature files found!")
        print("Please run feature extraction script first.")
        exit()
    
    print("\nAvailable features:")
    for i, feat in enumerate(available_features, 1):
        print(f"{i}. {feat}")
    
    # Select feature type
    if len(available_features) == 1:
        feature_type = available_features[0]
        print(f"\nUsing {feature_type} features")
    else:
        choice = input("\nSelect feature type (1/2): ").strip()
        feature_type = available_features[int(choice)-1]
    
    # Load features
    X_train, y_train, X_val, y_val, X_test, y_test = load_features(feature_type)
    
    # Preprocess features
    X_train_proc, X_val_proc, X_test_proc, scaler, pca = preprocess_features(
        X_train, X_val, X_test, use_pca=True
    )
    
    # Train and evaluate
    results_df, trained_models = train_and_evaluate(
        X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test, 
        scaler, pca, feature_type
    )
    results_df, trained_models = train_and_evaluate(
        X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test,
        scaler, pca, feature_type
    )
    
    print("\n" + "="*70)
    print("TRAINING AND CLASSIFICATION COMPLETED!")
    print("="*70)