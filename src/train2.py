import numpy as np
import pandas as pd
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

