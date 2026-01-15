import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Paths
DATA_DIR = "data/training"
RESULTS_DIR = "results"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_images_from_folder(folder_path):
    """Load all images from a folder"""
    images = []
    filenames = []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(np.array(img))
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return images, filenames

def explore_dataset():
    """Explore the dataset structure and properties"""
    print("="*70)
    print("DATASET EXPLORATION - PURE MACHINE LEARNING APPROACH")
    print("="*70)
    
    classes = ['rock', 'paper', 'scissors']
    class_data = {}
    
    # Load data for each class
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist!")
            continue
        
        images, filenames = load_images_from_folder(class_path)
        class_data[class_name] = {
            'images': images,
            'filenames': filenames,
            'count': len(images)
        }
        
        print(f"\n{class_name.upper()}:")
        print(f"  Number of images: {len(images)}")
        
        if len(images) > 0:
            # Analyze image properties
            shapes = [img.shape for img in images]
            heights = [s[0] for s in shapes]
            widths = [s[1] for s in shapes]
            
            print(f"  Image dimensions:")
            print(f"    Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")
            print(f"    Width:  min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
    
    return class_data

def visualize_class_distribution(class_data):
    """Visualize the distribution of samples across classes"""
    print("\n" + "="*70)
    print("GENERATING CLASS DISTRIBUTION PLOT")
    print("="*70)
    
    classes = list(class_data.keys())
    counts = [class_data[c]['count'] for c in classes]
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = plt.bar(classes, counts, color=colors, alpha=0.8)
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Images', fontsize=12, fontweight='bold')
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/class_distribution.png")
    plt.close()

def visualize_sample_images(class_data, samples_per_class=3):
    """Display sample images from each class"""
    print("\n" + "="*70)
    print("GENERATING SAMPLE IMAGES VISUALIZATION")
    print("="*70)
    
    classes = list(class_data.keys())
    
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(12, 4*len(classes)))
    
    for i, class_name in enumerate(classes):
        images = class_data[class_name]['images']
        
        # Select random samples
        num_samples = min(samples_per_class, len(images))
        indices = np.random.choice(len(images), num_samples, replace=False)
        
        for j in range(samples_per_class):
            if j < num_samples:
                img = images[indices[j]]
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{class_name.upper()}', 
                                        fontsize=12, fontweight='bold', loc='left')
            else:
                axes[i, j].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_images.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/sample_images.png")
    plt.close()

def analyze_image_statistics(class_data):
    """Analyze basic statistics of images"""
    print("\n" + "="*70)
    print("IMAGE STATISTICS ANALYSIS")
    print("="*70)
    
    for class_name, data in class_data.items():
        images = data['images']
        
        if len(images) == 0:
            continue
        
        print(f"\n{class_name.upper()}:")
        
        # Compute statistics on a sample
        sample_size = min(50, len(images))
        sample_images = images[:sample_size]
        
        # Brightness statistics
        brightness_values = [np.mean(img) for img in sample_images]
        print(f"  Brightness: min={min(brightness_values):.1f}, "
              f"max={max(brightness_values):.1f}, avg={np.mean(brightness_values):.1f}")
        
        # Color channel statistics
        red_means = [np.mean(img[:,:,0]) for img in sample_images]
        green_means = [np.mean(img[:,:,1]) for img in sample_images]
        blue_means = [np.mean(img[:,:,2]) for img in sample_images]
        
        print(f"  Red channel avg:   {np.mean(red_means):.1f}")
        print(f"  Green channel avg: {np.mean(green_means):.1f}")
        print(f"  Blue channel avg:  {np.mean(blue_means):.1f}")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("ROCK-PAPER-SCISSORS DETECTION")
    print("PURE MACHINE LEARNING APPROACH")
    print("="*70)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Data directory '{DATA_DIR}' not found!")
        print("Please ensure your data is organized as:")
        print("  data/rock/")
        print("  data/paper/")
        print("  data/scissors/")
        return
    
    # Explore dataset
    class_data = explore_dataset()
    
    if not class_data:
        print("\nError: No data found!")
        return
    
    # Generate visualizations
    visualize_class_distribution(class_data)
    visualize_sample_images(class_data)
    
    # Analyze statistics
    analyze_image_statistics(class_data)
    
    # Summary
    total_images = sum(data['count'] for data in class_data.values())
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images: {total_images}")
    print(f"Classes: {len(class_data)}")
    print(f"Visualizations saved in: {RESULTS_DIR}/")
    print("\nNext step: Run feature_extraction_media.py")
    print("="*70)

if __name__ == "__main__":
    main()