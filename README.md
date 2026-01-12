# Rock-Paper-Scissors-ML

A machine learning project that classifies hand gestures for Rock-Paper-Scissors game using computer vision and traditional ML algorithms.

## Overview

This project uses hand-crafted feature extraction techniques combined with classical machine learning algorithms to classify hand gestures into three categories: Rock, Paper, and Scissors. The approach focuses on geometric features, shape descriptors, and finger counting algorithms rather than deep learning.

## Dataset

The project uses the Rock Paper Scissors dataset from Kaggle:
**Dataset Link**: https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors

Download and organize the data as follows:
```
data/
├── rock/
├── paper/
└── scissors/
```

## Features Extracted

- **Geometric Features**: Convex hull, bounding box, area ratios
- **Shape Descriptors**: Hu moments (7 invariant features)
- **Hand-specific**: Finger counting, fist detection, centroid distance profiles
- **Total**: 26 features per image

## Machine Learning Models

- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors
- Support Vector Machine
- Artificial Neural Network

## Requirements

```bash
pip install numpy opencv-python pillow scikit-learn matplotlib seaborn pandas xgboost
```

## How to Run

### 1. Data Exploration
```bash
python src/data_exploration.py
```
Generates dataset statistics and visualizations in `results/` directory.

### 2. Feature Extraction
```bash
python src/feature_extraction.py
```
Extracts features from all images and saves them in `features/` directory.

### 3. Model Training
```bash
python src/train.py
```
Trains multiple ML models, evaluates performance, and saves the best model in `models/` directory.

### 4. Prediction
```bash
python src/predict.py
```
Use the trained model to predict on new images or test on folders.

## Project Structure

```
Rock-Paper-Scissors-ML/
├── src/
│   ├── data_exploration.py    # Dataset analysis
│   ├── feature_extraction.py  # Feature extraction pipeline
│   ├── train.py              # Model training and evaluation
│   └── predict.py            # Inference and prediction
├── data/                     # Dataset (download from Kaggle)
├── features/                 # Extracted features (generated)
├── models/                   # Trained models (generated)
├── results/                  # Visualizations and reports (generated)
└── README.md
```

## Results

The pipeline generates:
- Model performance comparisons
- Confusion matrices for each classifier
- Feature importance analysis
- Accuracy metrics and training times

All results are saved in the `results/` directory with detailed visualizations.