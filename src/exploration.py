import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"

def explore_data():
    splits = ["training", "validation", "testing"]
    gestures = ["rock", "paper", "scissors"]
    
    for split in splits:
        print(f"\n{split.upper()}:")
        split_total = 0
        counts = {}
        
        for gesture in gestures:
            if split == "validation":
                path = DATA_DIR / split
                count = len(list(path.glob(f"{gesture}*.png")))
            else:
                path = DATA_DIR / split / gesture
                count = len(list(path.glob("*.png")))
            counts[gesture] = count
            split_total += count
        
        for gesture in gestures:
            pct = (counts[gesture] / split_total * 100) if split_total > 0 else 0
            print(f"  {gesture}: {counts[gesture]} ({pct:.1f}%)")
        print(f"  Total: {split_total}")

def analyze_image_properties():    
    widths, heights = [], []
    for split in ["training", "testing"]:
        for gesture in ["rock", "paper", "scissors"]:
            path = DATA_DIR / split / gesture
            for img_path in path.glob("*.png"):
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
    
    for img_path in (DATA_DIR / "validation").glob("*.png"):
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)
    
    aspect_ratios = [w/h for w, h in zip(widths, heights)]
    
    print("\nIMAGE PROPERTIES:")
    print(f"  Width - Min: {min(widths)}, Max: {max(widths)}, Avg: {np.mean(widths):.1f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {np.mean(heights):.1f}")
    print(f"  Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Avg: {np.mean(aspect_ratios):.2f}")

if __name__ == "__main__":
    explore_data()
    analyze_image_properties()