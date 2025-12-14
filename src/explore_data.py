from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

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
                # Validation has files named like "rock01.png" in a single folder
                path = DATA_DIR / split
                if path.exists():
                    count = len(list(path.glob(f"{gesture}*.png")))
                else:
                    count = 0
            else:
                # Training and testing have subfolders
                path = DATA_DIR / split / gesture
                if path.exists():
                    count = len(list(path.glob("*.png")))
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
                for img_path in path.glob("*.png"):
                    img = Image.open(img_path)
                    widths.append(img.width)
                    heights.append(img.height)
    
    # Validation set (different structure)
    val_path = DATA_DIR / "validation"
    if val_path.exists():
        for img_path in val_path.glob("*.png"):
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
    
    aspect_ratios = [w/h for w, h in zip(widths, heights)]
    
    print(f"\nImage Dimensions:")
    print(f"  Width  - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.1f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.1f}")
    print(f"\nAspect Ratios:")
    print(f"  Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Avg: {np.mean(aspect_ratios):.2f}")
    print(f"\nRecommended target size: (224, 224)")
    
    return (224, 224)

def load_images_and_labels(split="training"):
    images, labels = [], []
    
    if split == "validation":
        # Validation has files named like "rock01.png" in a single folder
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split
            if path.exists():
                for img_path in sorted(path.glob(f"{gesture}*.png")):
                    img = Image.open(img_path)
                    images.append(np.array(img))
                    labels.append(label)
    else:
        # Training and testing have subfolders
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split / gesture
            if path.exists():
                for img_path in sorted(path.glob("*.png")):
                    img = Image.open(img_path)
                    images.append(np.array(img))
                    labels.append(label)
    
    return np.array(images), np.array(labels)

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
            image_files = list(path.glob("*.png"))[:6]
            
            for j, img_path in enumerate(image_files):
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{gesture.upper()}', 
                                        fontsize=12, fontweight='bold', 
                                        loc='left')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/sample_images.png', dpi=300, bbox_inches='tight')
    print("Sample images saved: results/plots/sample_images.png")
    plt.close()

if __name__ == "__main__":
    # Create results directory
    import os
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