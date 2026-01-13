import os
import hashlib
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from io import BytesIO

from src.path import DATA_PATH

class ImageDownloader:
    def __init__(self, save_dir_name="images", max_workers=16):
        # Save path: data/images
        self.save_dir = os.path.join(DATA_PATH, save_dir_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.max_workers = max_workers
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def _get_filename(self, url):
        """Generate MD5 filename from URL to ensure uniqueness."""
        hash_object = hashlib.md5(url.encode())
        return f"{hash_object.hexdigest()}.jpg"

    def _download_one(self, url):
        """Download and save a single image."""
        if not url or not isinstance(url, str):
            return None
        
        fname = self._get_filename(url)
        save_path = os.path.join(self.save_dir, fname)

        # Skip if already exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return save_path

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                # Convert to RGB (standardize)
                image.convert("RGB").save(save_path, "JPEG")
                return save_path
        except Exception:
            pass
        
        return None

    def _extract_url(self, item):
        """
        Helper to safely extract a single URL string from various input formats.
        """
        if item is None:
            return None
            
        # Case 1: Simple string
        if isinstance(item, str):
            return item
            
        # Case 2: List (take the first valid item)
        if isinstance(item, list) and len(item) > 0:
            first = item[0]
            if isinstance(first, str):
                return first
            elif isinstance(first, dict):
                # Try common keys for Amazon data
                return first.get('large_image_url') or first.get('hi_res') or first.get('image_url')
        
        # Case 3: Dictionary
        if isinstance(item, dict):
            return item.get('large_image_url') or item.get('hi_res') or item.get('image_url')
            
        return None

    def run(self, df: pd.DataFrame, url_col: str) -> pd.DataFrame:
        """
        Download images and add 'local_img_path' column to DataFrame.
        """
        print(f"[ImageDownloader] Downloading images to {self.save_dir}...")
        
        all_urls = []
        df_targets = df[url_col].apply(self._extract_url)
        unique_urls = list(set(df_targets.dropna().unique()))
        
        print(f"[ImageDownloader] Found {len(unique_urls)} unique images to download.")
        
        url_to_path = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use tqdm to show progress
            results = list(tqdm(executor.map(self._download_one, unique_urls), 
                                total=len(unique_urls), desc="Downloading"))
        
        for url, path in zip(unique_urls, results):
            url_to_path[url] = path

        df['local_img_path'] = df_targets.map(url_to_path)
        
        return df