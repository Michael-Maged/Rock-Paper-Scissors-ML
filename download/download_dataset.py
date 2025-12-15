"""
Script to download and prepare Rock Paper Scissors datasets from Kaggle

This script supports two popular datasets:
1. sanikamal/rock-paper-scissors-dataset (840 images)
2. glushko/rock-paper-scissors-dataset (2188 images)

Prerequisites:
1. Install Kaggle API: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)
   - Set permissions: chmod 600 ~/.kaggle/kaggle.json
"""

import shutil
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset information
DATASETS = {
    "sanikamal": {
        "name": "sanikamal/rock-paper-scissors-dataset",
        "description": "840 training images + validation set",
        "size": "~60MB"
    },
    "glushko": {
        "name": "glushko/rock-paper-scissors-dataset",
        "description": "2188 images (rock/paper/scissors)",
        "size": "~300MB"
    },
    "drgfreeman": {
        "name": "drgfreeman/rockpaperscissors",
        "description": "Rock Paper Scissors images dataset",
        "size": "~200MB"
    }
}

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    try:
        import importlib.util
        if importlib.util.find_spec("kaggle") is None:
            print("✗ Kaggle API not installed")
            print("  Install with: pip install kaggle")
            return False
        print("✓ Kaggle API is installed")
        return True
    except OSError :
        print("✗ Kaggle API credentials not found")
        print("  Please follow these steps:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows)")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

def download_dataset(dataset_key="sanikamal", max_retries=3):
    """Download dataset from Kaggle with retry logic for corrupted files"""
    
    if dataset_key not in DATASETS:
        print(f"Error: Unknown dataset '{dataset_key}'")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return None
    
    dataset_info = DATASETS[dataset_key]
    dataset_name = dataset_info["name"]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name}")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: {dataset_info['size']}")
    print(f"{'='*60}\n")
    
    # Create temp directory
    temp_dir = PROJECT_ROOT / "temp_download"
    temp_dir.mkdir(exist_ok=True)
    
    import time
    
    for attempt in range(1, max_retries + 1):
        try:
            # Clean up any corrupted zip files before downloading
            for zip_file in temp_dir.glob("*.zip"):
                try:
                    zip_file.unlink()
                    print(f"  Removed corrupted zip: {zip_file.name}")
                except Exception:
                    print(f"  Warning: Could not remove {zip_file.name}")
            
            print(f"Downloading dataset (attempt {attempt}/{max_retries})...")
            
            # Download dataset
            import kaggle
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(temp_dir),
                unzip=True
            )
            print("✓ Download complete")
            
            return temp_dir
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a zip file corruption error
            if "corrupted" in error_msg or "not a valid zip" in error_msg or "bad crc" in error_msg:
                if attempt < max_retries:
                    print("✗ Corrupted zip file detected. Retrying...")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    print(f"✗ Download failed after {max_retries} attempts - zip file was corrupted")
                    # Clean up corrupted files
                    for item in temp_dir.glob("*"):
                        try:
                            if item.is_file():
                                item.unlink()
                            else:
                                shutil.rmtree(item)
                        except Exception:
                            pass
                    return None
            else:
                print(f"✗ Error downloading dataset: {e}")
                return None
    
    return None

def organize_sanikamal_dataset(temp_dir):
    """Organize sanikamal dataset into project structure"""
    
    print("\nOrganizing dataset...")
    
    # This dataset typically has structure: Rock-Paper-Scissors/
    source_base = temp_dir / "Rock-Paper-Scissors"
    if not source_base.exists():
        # Try alternative structure
        source_base = temp_dir
    
    # Expected splits in sanikamal dataset
    splits_map = {
        "train": "training",
        "test": "testing",
        "validation": "validation"
    }
    
    gestures = ["rock", "paper", "scissors"]
    
    for source_split, target_split in splits_map.items():
        source_path = source_base / source_split
        
        if not source_path.exists():
            print(f"  Warning: {source_split} directory not found, skipping")
            continue
        
        print(f"  Processing {source_split} -> {target_split}")
        
        for gesture in gestures:
            source_gesture = source_path / gesture
            
            if target_split == "validation":
                # Validation has flat structure with prefixed filenames
                target_path = DATA_DIR / target_split
            else:
                # Training/testing have gesture subdirectories
                target_path = DATA_DIR / target_split / gesture
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            if source_gesture.exists():
                # Copy images
                images = list(source_gesture.glob("*.png")) + list(source_gesture.glob("*.jpg"))
                
                for img in images:
                    if target_split == "validation":
                        # Prefix with gesture name for validation
                        target_name = f"{gesture}_{img.name}"
                        shutil.copy2(img, target_path / target_name)
                    else:
                        shutil.copy2(img, target_path / img.name)
                
                print(f"    {gesture}: {len(images)} images")

def organize_glushko_dataset(temp_dir):
    """Organize glushko dataset into project structure"""
    
    print("\nOrganizing dataset...")
    
    # Debug: Show directory structure
    print(f"  Temp directory contents: {list(temp_dir.iterdir())}")
    
    # This dataset has structure: train/, val/, test/ with gesture subdirs
    gestures = ["rock", "paper", "scissors"]
    splits = ["train", "val", "test"]
    
    # Try to detect structure
    has_split_dirs = all((temp_dir / split).exists() for split in splits)
    
    if has_split_dirs:
        # Structure: train/, val/, test/ directories, each with rock/, paper/, scissors/ subdirs
        print("  Detected split-based structure (train/val/test)")
        organize_glushko_split_structure(temp_dir, gestures, splits)
    else:
        # Structure: rock/, paper/, scissors/ at root - need to split ourselves
        print("  Detected gesture-based structure (rock/paper/scissors)")
        organize_glushko_gesture_structure(temp_dir, gestures)

def organize_glushko_split_structure(temp_dir, gestures, splits):
    """Organize glushko dataset when it already has train/val/test splits"""
    
    split_map = {"train": "training", "val": "validation", "test": "testing"}
    
    for split_src, split_tgt in split_map.items():
        split_path = temp_dir / split_src
        
        if not split_path.exists():
            print(f"  Warning: {split_src} directory not found")
            continue
        
        print(f"\n  Processing {split_src} -> {split_tgt}")
        
        for gesture in gestures:
            gesture_path = split_path / gesture
            
            if not gesture_path.exists():
                print(f"    Warning: {gesture} not found in {split_src}")
                continue
            
            # Get all images
            images = list(gesture_path.glob("*.png")) + list(gesture_path.glob("*.jpg")) + list(gesture_path.glob("*.jpeg"))
            
            if split_tgt == "validation":
                # Validation has flat structure with prefixed filenames
                target_path = DATA_DIR / split_tgt
            else:
                # Training/testing have gesture subdirectories
                target_path = DATA_DIR / split_tgt / gesture
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            for img in images:
                if split_tgt == "validation":
                    target_name = f"{gesture}_{img.name}"
                    shutil.copy2(img, target_path / target_name)
                else:
                    shutil.copy2(img, target_path / img.name)
            
            print(f"    {gesture}: {len(images)} images")

def organize_glushko_gesture_structure(temp_dir, gestures):
    """Organize glushko dataset when it has gesture folders at root"""
    
    # Count total images
    total_images = {}
    for gesture in gestures:
        source_path = temp_dir / gesture
        if source_path.exists():
            images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
            total_images[gesture] = len(images)
    
    print(f"  Total images found: {sum(total_images.values())}")
    
    # Split ratios: 70% train, 15% validation, 15% test
    train_ratio = 0.70
    val_ratio = 0.15
    
    import random
    random.seed(42)
    
    for gesture in gestures:
        source_path = temp_dir / gesture
        
        if not source_path.exists():
            print(f"  Warning: {gesture} directory not found")
            continue
        
        # Get all images
        images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
        random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        print(f"\n  {gesture}:")
        print(f"    Total: {n_images}")
        print(f"    Training: {len(train_images)}")
        print(f"    Validation: {len(val_images)}")
        print(f"    Testing: {len(test_images)}")
        
        # Copy to training
        train_path = DATA_DIR / "training" / gesture
        train_path.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy2(img, train_path / img.name)
        
        # Copy to validation (flat structure with prefix)
        val_path = DATA_DIR / "validation"
        val_path.mkdir(parents=True, exist_ok=True)
        for img in val_images:
            target_name = f"{gesture}_{img.name}"
            shutil.copy2(img, val_path / target_name)
        
        # Copy to testing
        test_path = DATA_DIR / "testing" / gesture
        test_path.mkdir(parents=True, exist_ok=True)
        for img in test_images:
            shutil.copy2(img, test_path / img.name)

def organize_drgfreeman_dataset(temp_dir):
    """Organize drgfreeman dataset into project structure"""
    
    print("\nOrganizing dataset...")
    
    # Debug: Show directory structure
    print(f"  Temp directory contents: {list(temp_dir.iterdir())}")
    
    # Try to find the actual root by checking subdirectories
    root_dir = temp_dir
    
    # Check if there's a single subdirectory that contains the dataset
    subdirs = [d for d in temp_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        potential_root = subdirs[0]
        gestures_in_subdir = [d.name for d in potential_root.iterdir() if d.is_dir()]
        if any(g in gestures_in_subdir for g in ["rock", "paper", "scissors"]):
            root_dir = potential_root
            print(f"  Found dataset in subdirectory: {potential_root.name}")
    
    # This dataset typically has: rock/ paper/ scissors/ folders at root level
    gestures = ["rock", "paper", "scissors"]
    
    # Count total images
    total_images = {}
    for gesture in gestures:
        source_path = root_dir / gesture
        if source_path.exists():
            images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
            total_images[gesture] = len(images)
    
    print(f"  Total images found: {sum(total_images.values())}")
    
    # Split ratios: 70% train, 15% validation, 15% test
    train_ratio = 0.70
    val_ratio = 0.15
    
    import random
    random.seed(42)
    
    for gesture in gestures:
        source_path = root_dir / gesture
        
        if not source_path.exists():
            print(f"  Warning: {gesture} directory not found")
            continue
        
        # Get all images
        images = list(source_path.glob("*.png")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
        random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        print(f"\n  {gesture}:")
        print(f"    Total: {n_images}")
        print(f"    Training: {len(train_images)}")
        print(f"    Validation: {len(val_images)}")
        print(f"    Testing: {len(test_images)}")
        
        # Copy to training
        train_path = DATA_DIR / "training" / gesture
        train_path.mkdir(parents=True, exist_ok=True)
        for img in train_images:
            shutil.copy2(img, train_path / img.name)
        
        # Copy to validation (flat structure with prefix)
        val_path = DATA_DIR / "validation"
        val_path.mkdir(parents=True, exist_ok=True)
        for img in val_images:
            target_name = f"{gesture}_{img.name}"
            shutil.copy2(img, val_path / target_name)
        
        # Copy to testing
        test_path = DATA_DIR / "testing" / gesture
        test_path.mkdir(parents=True, exist_ok=True)
        for img in test_images:
            shutil.copy2(img, test_path / img.name)

def cleanup(temp_dir):
    """Remove temporary download directory"""
    if temp_dir and temp_dir.exists():
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")

def verify_structure():
    """Verify the data directory structure"""
    print(f"\n{'='*60}")
    print("Verifying data structure...")
    print(f"{'='*60}")
    
    splits = ["training", "validation", "testing"]
    gestures = ["rock", "paper", "scissors"]
    
    total_count = 0
    
    for split in splits:
        split_count = 0
        print(f"\n{split}:")
        
        for gesture in gestures:
            if split == "validation":
                path = DATA_DIR / split
                count = len(list(path.glob(f"{gesture}*.png"))) + len(list(path.glob(f"{gesture}*.jpg")))
            else:
                path = DATA_DIR / split / gesture
                if path.exists():
                    count = len(list(path.glob("*.png"))) + len(list(path.glob("*.jpg")))
                else:
                    count = 0
            
            print(f"  {gesture}: {count} images")
            split_count += count
        
        print(f"  Total: {split_count}")
        total_count += split_count
    
    print(f"\n{'='*60}")
    print(f"Grand Total: {total_count} images")
    print(f"{'='*60}\n")
    
    if total_count == 0:
        print("⚠ Warning: No images found! Dataset may not have been downloaded correctly.")
        return False
    
    return True

def display_menu():
    """Display dataset selection menu"""
    print("\n" + "="*60)
    print("ROCK PAPER SCISSORS DATASET SETUP")
    print("="*60)
    print("\nAvailable datasets:")
    print("\n1. Sanikamal Dataset")
    print("   - 840 training images + validation set")
    print("   - Size: ~60MB")
    print("\n2. Glushko Dataset")
    print("   - 2188 images (rock/paper/scissors)")
    print("   - Size: ~300MB")
    print("\n3. DrGFreeman Dataset")
    print("   - Rock Paper Scissors images dataset")
    print("   - Size: ~200MB")
    print("\n4. All Datasets")
    print("   - Downloads and combines all three datasets")
    print("   - Total size: ~560MB")
    print("\n" + "="*60)

def get_user_choice():
    """Get dataset choice from user"""
    display_menu()
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                return choice
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\n\nSetup cancelled by user.")
            exit(0)

def main():
    parser = argparse.ArgumentParser(description="Download and setup Rock Paper Scissors dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["1", "2", "3", "4"],
        default=None,
        help="Which dataset to download: 1=sanikamal, 2=glushko, 3=drgfreeman, 4=all (interactive if not specified)"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and only organize existing data in temp_download/"
    )
    
    args = parser.parse_args()
    
    # Get dataset choice
    if args.dataset:
        dataset_choice = args.dataset
    else:
        dataset_choice = get_user_choice()
    
    # Map choice to datasets
    datasets_to_process = []
    
    if dataset_choice == "4":
        datasets_to_process = ["sanikamal", "glushko", "drgfreeman"]
    else:
        dataset_map = {"1": "sanikamal", "2": "glushko", "3": "drgfreeman"}
        datasets_to_process = [dataset_map[dataset_choice]]
    
    print("\n" + "="*60)
    print("ROCK PAPER SCISSORS DATASET SETUP")
    print("="*60)
    
    # Check Kaggle setup
    if not args.skip_download and not check_kaggle_setup():
        return
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    for dataset_name in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*60}")
        
        # Download dataset
        if args.skip_download:
            temp_dir = PROJECT_ROOT / "temp_download"
            if not temp_dir.exists():
                print("Error: temp_download directory not found")
                continue
        else:
            temp_dir = download_dataset(dataset_name)
            if not temp_dir:
                continue
        
        # Organize dataset based on type
        try:
            if dataset_name == "sanikamal":
                organize_sanikamal_dataset(temp_dir)
            elif dataset_name == "glushko":
                organize_glushko_dataset(temp_dir)
            else:  # drgfreeman
                organize_drgfreeman_dataset(temp_dir)
            
            print(f"✓ {dataset_name} organized successfully!")
            
        finally:
            # Cleanup
            if not args.skip_download:
                cleanup(temp_dir)
    
    # Verify final structure
    if verify_structure():
        print("✓ Dataset setup complete!")
        print(f"✓ Data ready in: {DATA_DIR}")
        print("\nNext steps:")
        print("  1. Explore data: python src/explore_data.py")
        print("  2. Train model: python src/train_model.py")

if __name__ == "__main__":
    main()