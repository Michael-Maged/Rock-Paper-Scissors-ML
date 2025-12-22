from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

DATA_DIR = Path(__file__).parent.parent / "data"
LABEL_MAP = {"rock": 0, "paper": 1, "scissors": 2}

def explore_data():
    print("="*70)
    print("DATASET EXPLORATION")
    print("="*70)

    splits = ["training", "validation", "testing"]
    gestures = ["rock", "paper", "scissors"]

    all_counts = {}

    for split in splits:
        print(f"\n{split.upper()}:")
        split_total = 0
        counts = {}

        for gesture in gestures:
            if split == "validation":
                # Validation has flat structure
                path = DATA_DIR / "validation"
                if path.exists():
                    all_files = [f for f in path.glob("*.png") if gesture in f.name]
                    count = len(all_files)
                else:
                    count = 0
            else:
                path = DATA_DIR / split / gesture
                if path.exists():
                    all_files = list(path.glob("*.png"))
                    count = len(all_files)
                else:
                    count = 0

            counts[gesture] = count
            split_total += count

        for gesture in gestures:
            pct = (counts[gesture] / split_total * 100) if split_total > 0 else 0
            print(f"  {gesture}: {counts[gesture]:>4} ({pct:.1f}%)")
        print(f"  Total: {split_total:>4}")

        all_counts[split] = counts

    # Visualize class distribution
    visualize_class_distribution(all_counts)

    return all_counts

def visualize_class_distribution(all_counts):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    splits = ["training", "validation", "testing"]
    gestures = ["rock", "paper", "scissors"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, split in enumerate(splits):
        counts = all_counts[split]
        values = [counts[g] for g in gestures]
        
        axes[idx].bar(gestures, values, color=colors)
        axes[idx].set_title(f'{split.capitalize()} Set', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Number of Images', fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (gesture, count) in enumerate(zip(gestures, values)):
            axes[idx].text(i, count + 10, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nClass distribution plot saved: results/plots/class_distribution.png")
    plt.close()

def analyze_image_properties():
    print("\n" + "="*70)
    print("IMAGE PROPERTIES ANALYSIS")
    print("="*70)

    widths, heights = [], []

    # Training and testing sets
    for split in ["training", "testing"]:
        for gesture in ["rock", "paper", "scissors"]:
            path = DATA_DIR / split / gesture
            if path.exists():
                all_files = list(path.glob("*.png"))
                random.seed(42)
                random.shuffle(all_files)
                n = len(all_files)
                if split == "training":
                    files = all_files[:int(0.7 * n)]
                else:
                    files = all_files[int(0.7 * n) + int(0.15 * n):]
                for img_path in files[:50]:  # Sample 50 images per gesture per split
                    try:
                        img = Image.open(img_path)
                        widths.append(img.width)
                        heights.append(img.height)
                    except Exception as e:
                        print(f"Warning: Could not load {img_path}: {e}")

    # Validation set
    for gesture in ["rock", "paper", "scissors"]:
        path = DATA_DIR / "validation"
        if path.exists():
            all_files = [f for f in path.glob("*.png") if gesture in f.name]
            random.seed(42)
            random.shuffle(all_files)
            for img_path in all_files[:17]:  # Sample ~50 total
                try:
                    img = Image.open(img_path)
                    widths.append(img.width)
                    heights.append(img.height)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
    
    if widths:
        aspect_ratios = [w/h for w, h in zip(widths, heights)]
        
        print(f"\nImage Dimensions (from {len(widths)} samples):")
        print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.1f}")
        print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.1f}")
        print("\nAspect Ratios:")
        print(f"  Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Avg: {np.mean(aspect_ratios):.2f}")
    
    print("\nRecommended target size: (224, 224)")
    
    return (224, 224)

def load_images_and_labels(split="training"):
    images, labels = [], []

    if split == "validation":
        # Validation has flat structure with all images at root
        path = DATA_DIR / "validation"
        if path.exists():
            for gesture, label in LABEL_MAP.items():
                all_files = [f for f in path.glob("*.png") if gesture in f.name]
                for img_path in all_files:
                    try:
                        img = Image.open(img_path)
                        images.append(np.array(img))
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Could not load {img_path}: {e}")
    else:
        # Training and testing have subdirectories
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split / gesture
            if path.exists():
                all_files = list(path.glob("*.png"))
                random.seed(42)  # For reproducible splits
                random.shuffle(all_files)
                files = all_files
                for img_path in files:
                    try:
                        img = Image.open(img_path)
                        images.append(np.array(img))
                        labels.append(label)
                    except Exception as e:
                        print(f"Warning: Could not load {img_path}: {e}")

    return images, np.array(labels)

def visualize_sample_images():
    print("\n" + "="*70)
    print("VISUALIZING SAMPLE IMAGES")
    print("="*70)

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    gestures = ["rock", "paper", "scissors"]

    for i, gesture in enumerate(gestures):
        # Get images from training set
        path = DATA_DIR / "training" / gesture
        if path.exists():
            all_files = list(path.glob("*.png"))
            random.seed(42)
            random.shuffle(all_files)
            n = len(all_files)
            train_files = all_files[:int(0.7 * n)]
            image_files = train_files[:6]

            for j, img_path in enumerate(image_files):
                try:
                    img = Image.open(img_path)
                    axes[i, j].imshow(img)
                    axes[i, j].axis('off')
                    if j == 0:
                        axes[i, j].set_title(f'{gesture.upper()}',
                                            fontsize=12, fontweight='bold',
                                            loc='left')
                except Exception as e:
                    print(f"Warning: Could not display {img_path}: {e}")
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/sample_images.png', dpi=300, bbox_inches='tight')
    print("Sample images saved: results/plots/sample_images.png")
    plt.close()

if __name__ == "__main__":
    # Create results directory
    os.makedirs('results/plots', exist_ok=True)
    
    # Explore dataset
    all_counts = explore_data()
    
    # Analyze image properties
    target_size = analyze_image_properties()
    
    # Visualize samples
    visualize_sample_images()
    
    print("\n" + "="*70)
    print("DATA EXPLORATION COMPLETED!")
    print("="*70)