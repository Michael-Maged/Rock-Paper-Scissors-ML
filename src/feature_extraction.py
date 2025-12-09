import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data"
LABEL_MAP = {"rock": 0, "paper": 1, "scissors": 2}

def load_images_and_labels(split="training"):
    images, labels = [], []
    
    if split == "validation":
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split
            for img_path in path.glob(f"{gesture}*.png"):
                img = Image.open(img_path)
                images.append(np.array(img))
                labels.append(label)
    else:
        for gesture, label in LABEL_MAP.items():
            path = DATA_DIR / split / gesture
            for img_path in path.glob("*.png"):
                img = Image.open(img_path)
                images.append(np.array(img))
                labels.append(label)
    
    return np.array(images), np.array(labels)

