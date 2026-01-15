# Rock-Paper-Scissors-ML

A machine learning project that classifies hand gestures for Rock-Paper-Scissors game using computer vision and traditional ML algorithms.

## Overview

This project implements **two different approaches** to classify hand gestures:

1. **Geometric Features Approach** (`src/`): Hand-crafted features using OpenCV (convex hull, contours, Hu moments, finger counting)
2. **MediaPipe Landmarks Approach** (`mediapipe/`): Hand landmarks detection using Google's MediaPipe library

Both approaches use classical machine learning algorithms (Decision Tree, Random Forest, XGBoost, KNN, SVM, ANN) rather than deep learning.

## Dataset

The project uses the Rock Paper Scissors dataset from Kaggle:
**Dataset Link**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Download and organize the data as follows:
```
data/
├── training/
│   ├── rock/
│   ├── paper/
│   └── scissors/
├── testing/
│   ├── rock/
│   ├── paper/
│   └── scissors/
└── predict/          # Your own test images
```

## Features Extracted

### Approach 1: Geometric Features (`src/`)
- **Geometric Features**: Convex hull, bounding box, area ratios
- **Shape Descriptors**: Hu moments (7 invariant features)
- **Hand-specific**: Finger counting, fist detection, centroid distance profiles
- **Total**: 23 features per image

### Approach 2: MediaPipe Landmarks (`mediapipe/`)
- **Hand Landmarks**: 21 landmarks × 3 coordinates (x, y, z) = 63 features
- **Relative Distances**: 15 distance features between key points
- **Finger Angles**: 5 angle features for finger bending
- **Total**: 83 features per image

## Machine Learning Models

- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors
- Support Vector Machine
- Artificial Neural Network

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Rock-Paper-Scissors-ML.git
cd Rock-Paper-Scissors-ML
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Command Prompt - RECOMMENDED):
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# PowerShell users: First run this to allow scripts:
 Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
 venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# OR install manually:
# For Geometric Features approach only:
pip install numpy opencv-python pillow scikit-learn matplotlib seaborn pandas xgboost

# For MediaPipe approach (includes all above):
# IMPORTANT: If you have TensorFlow, uninstall it first: pip uninstall tensorflow
pip install numpy opencv-python pillow scikit-learn matplotlib seaborn pandas xgboost mediapipe==0.10.13
```

## Requirements

### Basic Requirements (for Geometric Features approach)
```bash
pip install numpy opencv-python pillow scikit-learn matplotlib seaborn pandas xgboost
```

### Additional for MediaPipe approach

**⚠️ IMPORTANT: MediaPipe Installation Instructions**

MediaPipe has dependency conflicts with TensorFlow. Follow these steps:

```bash
# If you have TensorFlow installed, uninstall it first:
pip uninstall tensorflow

# Install specific MediaPipe version that works:
pip install mediapipe==0.10.13
```

**Alternative: Install all at once**
```bash
pip install -r requirements.txt
```

**Troubleshooting MediaPipe:**
- If you get protobuf errors, run: `pip uninstall tensorflow` first
- Use MediaPipe version 0.10.13 for best compatibility
- Test installation: `python -c "import mediapipe as mp; print(mp.__version__)"`

## How to Run

### Approach 1: Geometric Features (OpenCV)

#### 1. Data Exploration
```bash
python src/data_exploration.py
```
Generates dataset statistics and visualizations in `results/` directory.

#### 2. Feature Extraction
```bash
python src/feature_extraction.py
```
Extracts geometric features from all images and saves them in `features/` directory.

#### 3. Model Training
```bash
python src/train.py
```
Trains multiple ML models, evaluates performance, and saves the best model in `models/` directory.

#### 4. Prediction
```bash
python src/predict.py
```
Use the trained model to predict on new images or test on folders.

---

### Approach 2: MediaPipe Landmarks

#### 1. Data Exploration
```bash
python mediapipe/data_exploration_media.py
```

#### 2. Feature Extraction
```bash
python mediapipe/feature_extraction_media.py
```
Extracts hand landmarks using MediaPipe and saves them in `features/` directory.

#### 3. Model Training
```bash
python mediapipe/train_media.py
```

#### 4. Prediction
```bash
python mediapipe/predict_media.py
```
Use the trained model with MediaPipe hand detection for predictions.

## Project Structure

```
Rock-Paper-Scissors-ML/
├── src/                          # Geometric features approach
│   ├── data_exploration.py       # Dataset analysis
│   ├── feature_extraction.py     # OpenCV feature extraction
│   ├── train.py                  # Model training
│   └── predict.py                # Prediction
├── mediapipe/                    # MediaPipe landmarks approach
│   ├── data_exploration_media.py # Dataset analysis
│   ├── feature_extraction_media.py # MediaPipe feature extraction
│   ├── train_media.py            # Model training
│   └── predict_media.py          # Prediction with MediaPipe
├── data/                         # Dataset (download from Kaggle)
│   ├── training/                 # Training images
│   │   ├── rock/
│   │   ├── paper/
│   │   └── scissors/
│   ├── testing/                  # Testing images
│   └── predict/                  # Your test images
├── features/                     # Extracted features (generated)
├── models/                       # Trained models (generated)
├── results/                      # Visualizations and reports (generated)
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
└── README.md
```

## Results

Both pipelines generate:
- Model performance comparisons
- Confusion matrices for each classifier
- Accuracy metrics and training times
- Precision, Recall, and F1-Score comparisons

All results are saved in the `results/` directory with detailed visualizations.

## Comparison: Geometric vs MediaPipe

| Feature | Geometric Approach | MediaPipe Approach |
|---------|-------------------|--------------------|
| **Features** | 23 hand-crafted features | 83 landmark-based features |
| **Dependencies** | OpenCV only | OpenCV + MediaPipe |
| **Speed** | Fast | Moderate (landmark detection) |
| **Accuracy** | Good for clear images | Better for varied poses |
| **Robustness** | Sensitive to background | More robust to background |

## Notes

- The `data/validation/` folder contains additional test images
- Both approaches save models to the same `models/` directory (run separately to avoid overwriting)
- MediaPipe approach typically achieves higher accuracy due to more precise hand detection
- Geometric approach is faster and requires fewer dependencies