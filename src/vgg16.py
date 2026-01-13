import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List

class VGG16Extractor:
    """
    Extracts visual features (4096-dim) from local image files using VGG-16 (fc2 layer).
    """

    def __init__(self, use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[VGG16Extractor] Loading VGG16 on {self.device}...")

        # Load VGG16 with ImageNet weights
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Reconstruct the model to output the first fully-connected layer (fc2, 4096-dim)
        # Original Classifier: 
        # (0): Linear(25088, 4096) -> (1): ReLU -> (2): Dropout -> (3): Linear(4096, 4096) ...
        # We keep up to layer 4 to get the 4096 features.
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(*list(vgg.classifier.children())[:4]) 
        
        self.model = nn.Sequential(
            self.features,
            self.avgpool,
            nn.Flatten(),
            self.classifier
        ).to(self.device).eval()

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _encode_one(self, img_path: str):
        """Read a local image file and extract features."""
        if not img_path or not os.path.exists(img_path):
            # Return zero vector if image is missing
            return np.zeros(4096, dtype=np.float32)
        
        try:
            # Open image and convert to RGB (handle PNG/Grayscale)
            image = Image.open(img_path).convert("RGB")
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feat = self.model(img_t)
            
            return feat.cpu().numpy().flatten()
        except Exception as e:
            # print(f"Error processing {img_path}: {e}")
            return np.zeros(4096, dtype=np.float32)

    def run(self, df: pd.DataFrame, path_col: str, output_col: str = "img_central") -> pd.DataFrame:
        """
        Process the DataFrame.
        path_col: Column containing local file paths (str).
        """
        if path_col not in df.columns:
            raise KeyError(f"{path_col} not in DataFrame")

        df = df.copy()
        feats = []
        
        paths = df[path_col].tolist()
        for path in tqdm(paths, desc="VGG16 Feature"):
            feats.append(self._encode_one(path))
            
        df[output_col] = feats
        return df