from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif
import joblib
import seaborn as sns
import xgboost as xgb

# Import from data loading module
from feature_extraction import extract_features_handcrafted, LABEL_MAP

MODEL_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Create directories if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def perform_feature_selection(X_train, y_train, X_val, X_test, method='all'):
    """
    Perform feature selection to identify most efficient features
    Methods: SelectKBest, RFE (Recursive Feature Elimination), Mutual Information
    """
    
    print("\n" + "="*60)
    print("FEATURE SELECTION")
    print("="*60)
    
    results = {}
    
    # Method 1: SelectKBest with ANOVA F-test
    print("\n--- Method 1: SelectKBest (ANOVA F-test) ---")
    k_features = min(100, X_train.shape[1])  # Select top 100 features or all if less
    selector_kbest = SelectKBest(score_func=f_classif, k=k_features)
    X_train_kbest = selector_kbest.fit_transform(X_train, y_train)
    X_val_kbest = selector_kbest.transform(X_val)
    X_test_kbest = selector_kbest.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_kbest.shape[1]}")
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature_Index': range(X_train.shape[1]),
        'Score': selector_kbest.scores_
    }).sort_values('Score', ascending=False)
    
    print("\nTop 10 features by F-score:")
    print(feature_scores.head(10))
    
    results['kbest'] = {
        'X_train': X_train_kbest,
        'X_val': X_val_kbest,
        'X_test': X_test_kbest,
        'selector': selector_kbest,
        'n_features': X_train_kbest.shape[1]
    }
    
    # Method 2: Recursive Feature Elimination (RFE) with Random Forest
    print("\n--- Method 2: RFE with Random Forest ---")
    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector_rfe = RFE(estimator=rf_estimator, n_features_to_select=k_features, step=10)
    X_train_rfe = selector_rfe.fit_transform(X_train, y_train)
    X_val_rfe = selector_rfe.transform(X_val)
    X_test_rfe = selector_rfe.transform(X_test)
    
    print(f"Selected features: {X_train_rfe.shape[1]}")
    print(f"Feature ranking (1=selected): {np.sum(selector_rfe.ranking_ == 1)} features")
    
    results['rfe'] = {
        'X_train': X_train_rfe,
        'X_val': X_val_rfe,
        'X_test': X_test_rfe,
        'selector': selector_rfe,
        'n_features': X_train_rfe.shape[1]
    }
    
    # Method 3: Mutual Information
    print("\n--- Method 3: Mutual Information ---")
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_train_mi = selector_mi.fit_transform(X_train, y_train)
    X_val_mi = selector_mi.transform(X_val)
    X_test_mi = selector_mi.transform(X_test)
    
    print(f"Selected features: {X_train_mi.shape[1]}")
    
    mi_scores = pd.DataFrame({
        'Feature_Index': range(X_train.shape[1]),
        'MI_Score': selector_mi.scores_
    }).sort_values('MI_Score', ascending=False)
    
    print("\nTop 10 features by Mutual Information:")
    print(mi_scores.head(10))
    
    results['mi'] = {
        'X_train': X_train_mi,
        'X_val': X_val_mi,
        'X_test': X_test_mi,
        'selector': selector_mi,
        'n_features': X_train_mi.shape[1]
    }
    
    # Save feature selection results
    joblib.dump(results, MODEL_DIR / "feature_selectors.pkl")
    
    # Plot feature importance comparison
    plot_feature_importance(feature_scores, mi_scores)
    
    return results

def plot_feature_importance(f_scores, mi_scores):
    """Plot feature importance from different methods"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 20 F-scores
    top_f = f_scores.head(20)
    ax1.barh(range(len(top_f)), top_f['Score'])
    ax1.set_yticks(range(len(top_f)))
    ax1.set_yticklabels([f"F{i}" for i in top_f['Feature_Index']])
    ax1.set_xlabel('F-Score')
    ax1.set_title('Top 20 Features by ANOVA F-test')
    ax1.invert_yaxis()
    
    # Top 20 MI scores
    top_mi = mi_scores.head(20)
    ax2.barh(range(len(top_mi)), top_mi['MI_Score'])
    ax2.set_yticks(range(len(top_mi)))
    ax2.set_yticklabels([f"F{i}" for i in top_mi['Feature_Index']])
    ax2.set_xlabel('Mutual Information Score')
    ax2.set_title('Top 20 Features by Mutual Information')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=150)
    print(f"\nFeature importance plot saved to {RESULTS_DIR / 'feature_importance.png'}")
    plt.close()

def train_all_classifiers(X_train, y_train, X_val, y_val, X_test, y_test, feature_set_name="all"):
    """Train multiple classifiers as required"""
    
    print(f"\n{'='*60}")
    print(f"TRAINING CLASSIFIERS - Feature Set: {feature_set_name.upper()}")
    print(f"{'='*60}")
    
    # Define all required classifiers
    classifiers = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42,
            probability=True
        ),
        "ANN (MLP)": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
    }
    
    results = {}
    metrics_data = []
    
    for name, clf in classifiers.items():
        print(f"\n--- Training {name} ---")
        
        # Train
        clf.fit(X_train, y_train)
        
        # Predictions
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        test_precision = precision_score(y_test, test_pred, average='weighted')
        test_recall = recall_score(y_test, test_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        print(f"Test Precision:      {test_precision:.4f}")
        print(f"Test Recall:         {test_recall:.4f}")
        print(f"Test F1-Score:       {test_f1:.4f}")
        
        results[name] = {
            "model": clf,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "val_pred": val_pred,
            "test_pred": test_pred
        }
        
        metrics_data.append({
            'Classifier': name,
            'Training_Acc': train_acc,
            'Validation_Acc': val_acc,
            'Test_Acc': test_acc,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1_Score': test_f1
        })
    
    # Create metrics comparison table
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df.sort_values('Test_Acc', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON - {feature_set_name.upper()}")
    print(f"{'='*60}")
    print(metrics_df.to_string(index=False))
    
    # Save metrics to CSV
    metrics_df.to_csv(RESULTS_DIR / f'classifier_comparison_{feature_set_name}.csv', index=False)
    
    return results, metrics_df

def generate_comprehensive_evaluation(results, y_val, y_test, feature_set_name="all"):
    """Generate comprehensive evaluation report with all metrics"""
    
    label_names = ["Rock", "Paper", "Scissors"]
    
    # Create detailed report for each classifier
    report_file = RESULTS_DIR / f'detailed_report_{feature_set_name}.txt'
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"DETAILED CLASSIFICATION REPORT - {feature_set_name.upper()}\n")
        f.write("="*60 + "\n\n")
        
        for name, result in results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{name}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write("VALIDATION SET:\n")
            f.write(classification_report(y_val, result['val_pred'], target_names=label_names))
            f.write("\n\nTEST SET:\n")
            f.write(classification_report(y_test, result['test_pred'], target_names=label_names))
            f.write("\n\n")
    
    print(f"\nDetailed report saved to {report_file}")
    
    # Generate confusion matrices for all classifiers
    n_classifiers = len(results)
    n_cols = 3
    n_rows = (n_classifiers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_classifiers > 1 else [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['test_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {result["test_acc"]:.4f}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    # Hide extra subplots
    for idx in range(n_classifiers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'all_confusion_matrices_{feature_set_name}.png', dpi=150)
    print(f"Confusion matrices saved to {RESULTS_DIR / f'all_confusion_matrices_{feature_set_name}.png'}")
    plt.close()

def plot_metrics_comparison(metrics_df, feature_set_name="all"):
    """Plot comprehensive metrics comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    classifiers = metrics_df['Classifier'].values
    x = np.arange(len(classifiers))
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    width = 0.25
    ax1.bar(x - width, metrics_df['Training_Acc'], width, label='Training', alpha=0.8)
    ax1.bar(x, metrics_df['Validation_Acc'], width, label='Validation', alpha=0.8)
    ax1.bar(x + width, metrics_df['Test_Acc'], width, label='Test', alpha=0.8)
    ax1.set_xlabel('Classifier')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison Across Splits')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classifiers, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Test metrics (Precision, Recall, F1)
    ax2 = axes[0, 1]
    width = 0.25
    ax2.bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
    ax2.bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
    ax2.bar(x + width, metrics_df['F1_Score'], width, label='F1-Score', alpha=0.8)
    ax2.set_xlabel('Classifier')
    ax2.set_ylabel('Score')
    ax2.set_title('Test Set Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classifiers, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Plot 3: Test Accuracy ranking
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(classifiers)))
    bars = ax3.barh(classifiers, metrics_df['Test_Acc'], color=colors)
    ax3.set_xlabel('Test Accuracy')
    ax3.set_title('Test Accuracy Ranking')
    ax3.set_xlim([0, 1.1])
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    ax3.invert_yaxis()
    
    # Plot 4: Performance metrics heatmap
    ax4 = axes[1, 1]
    metrics_for_heatmap = metrics_df[['Classifier', 'Test_Acc', 'Precision', 'Recall', 'F1_Score']].set_index('Classifier')
    sns.heatmap(metrics_for_heatmap.T, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax4, cbar_kws={'label': 'Score'})
    ax4.set_title('Test Metrics Heatmap')
    ax4.set_xlabel('Classifier')
    ax4.set_ylabel('Metric')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'comprehensive_comparison_{feature_set_name}.png', dpi=150)
    print(f"Comprehensive comparison saved to {RESULTS_DIR / f'comprehensive_comparison_{feature_set_name}.png'}")
    plt.close()

def main():
    """Main training pipeline following project requirements"""
    
    print("="*60)
    print("ROCK PAPER SCISSORS CLASSIFICATION PROJECT")
    print("Following Academic Requirements:")
    print("I.   Dataset split into training/testing")
    print("II.  Feature extraction")
    print("III. Feature selection")
    print("IV.  Multiple classifiers (DT, RF, XGBoost, KNN, ANN, SVM)")
    print("V.   Performance evaluation (accuracy, confusion matrix, etc.)")
    print("="*60)
    
    # Step I: Load Dataset (already split into training/validation/testing)
    print("\n" + "="*60)
    print("STEP I: LOADING DATASET")
    print("="*60)
    
    print("\nLoading training data...")
    X_train, y_train = extract_features_handcrafted("training")
    print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
    
    print("\nLoading validation data...")
    X_val, y_val = extract_features_handcrafted("validation")
    print(f"Validation samples: {len(X_val)}")
    
    print("\nLoading test data...")
    X_test, y_test = extract_features_handcrafted("testing")
    print(f"Test samples: {len(X_test)}")
    
    # Step II: Features already extracted in load_data.py
    # (HOG, color histograms, edge density)
    print("\n" + "="*60)
    print("STEP II: FEATURE EXTRACTION - COMPLETE")
    print("Features: HOG, Color Histograms, Edge Density")
    print("="*60)
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    # Step III: Feature Selection
    print("\n" + "="*60)
    print("STEP III: FEATURE SELECTION")
    print("="*60)
    
    feature_sets = perform_feature_selection(
        X_train_scaled, y_train,
        X_val_scaled, X_test_scaled
    )
    
    # Step IV & V: Train classifiers and evaluate
    all_results = {}
    all_metrics = {}
    
    # Train with original features (all features)
    print("\n" + "="*60)
    print("TRAINING WITH ALL FEATURES")
    print("="*60)
    results_all, metrics_all = train_all_classifiers(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        feature_set_name="all_features"
    )
    generate_comprehensive_evaluation(results_all, y_val, y_test, "all_features")
    plot_metrics_comparison(metrics_all, "all_features")
    all_results['all_features'] = results_all
    all_metrics['all_features'] = metrics_all
    
    # Train with SelectKBest features
    print("\n" + "="*60)
    print("TRAINING WITH SELECTKBEST FEATURES")
    print("="*60)
    results_kbest, metrics_kbest = train_all_classifiers(
        feature_sets['kbest']['X_train'], y_train,
        feature_sets['kbest']['X_val'], y_val,
        feature_sets['kbest']['X_test'], y_test,
        feature_set_name="kbest"
    )
    generate_comprehensive_evaluation(results_kbest, y_val, y_test, "kbest")
    plot_metrics_comparison(metrics_kbest, "kbest")
    all_results['kbest'] = results_kbest
    all_metrics['kbest'] = metrics_kbest
    
    # Find and save best model overall
    best_acc = 0
    best_model_name = ""
    best_feature_set = ""
    best_model = None
    
    for fs_name, results in all_results.items():
        for clf_name, result in results.items():
            if result['test_acc'] > best_acc:
                best_acc = result['test_acc']
                best_model_name = clf_name
                best_feature_set = fs_name
                best_model = result['model']
    
    # Save best model
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    
    # Save corresponding feature selector
    if best_feature_set != 'all_features':
        joblib.dump(feature_sets[best_feature_set]['selector'], 
                   MODEL_DIR / "best_feature_selector.pkl")
    
    # Final summary
    print("\n" + "="*60)
    print("PROJECT COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"Feature Set: {best_feature_set}")
    print(f"Test Accuracy: {best_acc:.4f}")
    print(f"\nModels saved in: {MODEL_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - Classifier comparison tables (CSV)")
    print("  - Confusion matrices for all models")
    print("  - Comprehensive metrics comparison")
    print("  - Feature importance plots")
    print("  - Detailed classification reports")
    
    return all_results, all_metrics

if __name__ == "__main__":
    results, metrics = main()