import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
import textstat

# ============================================================
# 1. Text Peripheral Features
# ============================================================
def analyze_text_peripheral(text: str):
    """
    Extracts text peripheral cues:
    1. Sentiment Polarity
    2. Subjectivity
    3. Readability (Flesch Reading Ease)
    4. Extremity (Absolute Polarity)
    """
    if not isinstance(text, str) or not text.strip():
        return [0.0, 0.0, 0.0, 0.0]
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    try:
        readability = textstat.flesch_reading_ease(text)
    except:
        readability = 0.0
        
    extremity = abs(polarity)
    
    return [polarity, subjectivity, readability, extremity]


# ============================================================
# 2. Image Peripheral Features
# ============================================================
def analyze_image_peripheral(img_path: str):
    """
    Extracts image peripheral cues based on User's original logic:
    1. Brightness (Mean of Grayscale)
    2. Contrast (Std of Grayscale)
    3. Saturation (Mean of S channel in HSV)
    4. Edge Intensity (Mean of Canny edges)
    """
    # Return zeros if path is invalid
    if not img_path or not os.path.exists(img_path):
        return [0.0, 0.0, 0.0, 0.0]
    
    try:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            return [0.0, 0.0, 0.0, 0.0]
        
        # 1. Brightness & 2. Contrast (Based on Grayscale)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image) / 255.0  # Normalize to 0-1
        contrast = gray_image.std() / 255.0       # Normalize to 0-1
        
        # 3. Saturation (Based on HSV S-channel)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv_image[:, :, 1]
        avg_saturation = np.mean(saturation) / 255.0 # Normalize to 0-1
        
        # 4. Edge Intensity (Based on Canny)
        edges = cv2.Canny(gray_image, 100, 200)
        edge_intensity = np.mean(edges) / 255.0      # Normalize to 0-1
        
        return [brightness, contrast, avg_saturation, edge_intensity]
    
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


# ============================================================
# 3. Pipeline Function
# ============================================================
def add_features(df: pd.DataFrame, text_col: str, path_col: str):
    df = df.copy()
    
    # 1. Text
    print("[Peripheral] Extracting Text Features...")
    df[text_col] = df[text_col].fillna("").astype(str)
    df['text_peripheral'] = df[text_col].apply(analyze_text_peripheral)
    
    # 2. Image (from local path)
    print("[Peripheral] Extracting Image Features from local files...")
    tqdm.pandas(desc="Image Peripheral")
    df['image_peripheral'] = df[path_col].progress_apply(analyze_image_peripheral)
    
    return df