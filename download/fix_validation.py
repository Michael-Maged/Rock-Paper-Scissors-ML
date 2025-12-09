from pathlib import Path
import shutil
import random

DATA_DIR = Path("data")
random.seed(42)

print("="*60)
print("FIXING VALIDATION SPLIT")
print("="*60)

# First, check what we have
print("\nCurrent structure:")
for split in ["training", "testing", "validation"]:
    split_path = DATA_DIR / split
    if split_path.exists():
        if split == "validation":
            # Count files directly in validation
            files = list(split_path.glob("*.png")) + list(split_path.glob("*.jpg"))
            print(f"{split}: {len(files)} files")
        else:
            # Count files in gesture subfolders
            for gesture in ["rock", "paper", "scissors"]:
                gesture_path = split_path / gesture
                if gesture_path.exists():
                    files = list(gesture_path.glob("*.png")) + list(gesture_path.glob("*.jpg"))
                    print(f"{split}/{gesture}: {len(files)} files")

# Create validation folder
val_path = DATA_DIR / "validation"
val_path.mkdir(exist_ok=True)

print("\n" + "="*60)
print("Creating validation split from test set...")
print("="*60)

gestures = ["rock", "paper", "scissors"]

for gesture in gestures:
    test_path = DATA_DIR / "testing" / gesture
    
    if not test_path.exists():
        print(f"⚠ Warning: {test_path} doesn't exist")
        continue
    
    # Get all test images
    images = list(test_path.glob("*.png")) + list(test_path.glob("*.jpg"))
    print(f"\n{gesture}: found {len(images)} test images")
    
    if len(images) == 0:
        print("  ⚠ No images found!")
        continue
    
    # Take 30% for validation
    n_val = int(len(images) * 0.3)
    random.shuffle(images)
    val_images = images[:n_val]
    
    # Move to validation with prefix
    moved = 0
    for img in val_images:
        new_name = f"{gesture}_{img.name}"
        try:
            shutil.move(str(img), str(val_path / new_name))
            moved += 1
        except Exception as e:
            print(f"  Error moving {img.name}: {e}")
    
    print(f"  ✓ Moved {moved} images to validation")
    print(f"  ✓ Remaining in test: {len(images) - moved}")

print("\n" + "="*60)
print("FINAL STRUCTURE:")
print("="*60)

# Show final counts
for split in ["training", "validation", "testing"]:
    split_path = DATA_DIR / split
    print(f"\n{split.upper()}:")
    
    if split == "validation":
        for gesture in gestures:
            files = list(split_path.glob(f"{gesture}*.png")) + list(split_path.glob(f"{gesture}*.jpg"))
            print(f"  {gesture}: {len(files)} images")
    else:
        for gesture in gestures:
            gesture_path = split_path / gesture
            if gesture_path.exists():
                files = list(gesture_path.glob("*.png")) + list(gesture_path.glob("*.jpg"))
                print(f"  {gesture}: {len(files)} images")

print("\n✓ Done!")