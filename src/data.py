import os
import gzip
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.path import RAW_PATH, PROCESSED_PATH
from src.utils import save_parquet

from .image_manager import ImageDownloader
from .bert import BertExtractor
from .vgg16 import VGG16Extractor
from .peripheral_features import add_features as add_peripheral

class DataLoader:
    def __init__(
        self,
        fname: str,
        raw_ext: str = "jsonl.gz",
        test_size: float = 0.2,
        val_size: float = 0.1,
        text_col: str = "clean_text",
        img_col: str = "large_image_url",
        vote_col: str = "vote",
        device: str = "cuda"
    ):
        self.fname = fname
        self.raw_path = os.path.join(RAW_PATH, f"{fname}.{raw_ext}")
        self.test_size = test_size
        self.val_size = val_size
        self.text_col = text_col
        self.img_col = img_col
        self.vote_col = vote_col
        self.device = device
        
        # Defines output paths
        self.train_path = os.path.join(PROCESSED_PATH, f"{fname}_train.parquet")
        self.val_path = os.path.join(PROCESSED_PATH, f"{fname}_val.parquet")
        self.test_path = os.path.join(PROCESSED_PATH, f"{fname}_test.parquet")

        # STEP 0: Check if data already exists
        if self._check_processed_exists():
            print(f"[DataLoader] ✅ Found existing processed data for '{fname}'.")
            print(f"             Skipping download & feature extraction to save time.")
            return  # Stop initialization here!

        print(f"[DataLoader] No existing data found. Starting full processing pipeline...")

        # Initialize Extractors (Only if data is NOT found)
        self.downloader = ImageDownloader(save_dir_name=f"{fname}_images")
        self.bert = BertExtractor(use_gpu=("cuda" in device))
        self.vgg = VGG16Extractor(use_gpu=("cuda" in device))
        
        # Start pipeline
        self.process()

    def _check_processed_exists(self) -> bool:
        """
        Returns True if Train/Val/Test parquet files already exist.
        """
        return (
            os.path.exists(self.train_path) and 
            os.path.exists(self.val_path) and 
            os.path.exists(self.test_path)
        )

    def load_raw(self) -> pd.DataFrame:
        print(f"[DataLoader] Loading raw data: {self.raw_path}")
        data = []
        try:
            with gzip.open(self.raw_path, 'rb') as f:
                for line in f:
                    data.append(json.loads(line))
        except json.JSONDecodeError:
            with gzip.open(self.raw_path, 'rb') as f:
                data = json.load(f)
        return pd.DataFrame(data)

    def _process_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[DataLoader] Processing label column: '{self.vote_col}'")
        
        if self.vote_col not in df.columns:
            if 'helpful_vote' in df.columns:
                self.vote_col = 'helpful_vote'
            elif 'votes' in df.columns:
                self.vote_col = 'votes'
            else:
                raise KeyError(f"Label column '{self.vote_col}' not found.")

        def clean_vote(val):
            if pd.isna(val) or val == "":
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                val = val.replace(',', '').strip()
                try:
                    return float(val)
                except:
                    return 0.0
            return 0.0

        raw_votes = df[self.vote_col].apply(clean_vote)
        
        df['temp_raw_vote'] = raw_votes
        
        initial_len = len(df)
        df = df[df['temp_raw_vote'] > 0].copy()
        dropped_len = initial_len - len(df)
        
        print(f"   - Filter Rule: Keep only votes > 0")
        print(f"   - Dropped {dropped_len} rows ({(dropped_len/initial_len)*100:.1f}%) with 0 votes.")
        print(f"   - Remaining rows: {len(df)}")

        if len(df) == 0:
            raise ValueError("No data left after filtering! Check your dataset.")

        # Log Transformation: log(x + 1)
        df['label'] = np.log1p(df['temp_raw_vote'])
        
        df.drop(columns=['temp_raw_vote'], inplace=True)

        print(f"   - Min Label: {df['label'].min():.4f}")
        print(f"   - Max Label: {df['label'].max():.4f}")
        
        return df

    def process(self):
        # 1. Load Data
        df = self.load_raw()
        print(f"[DataLoader] Loaded {len(df)} rows.")
        
        # 2. Process Labels (Filter & Log)
        df = self._process_labels(df)
        
        # 3. Download Images
        df = self.downloader.run(df, url_col=self.img_col)
        
        # Filter download failures
        initial_len = len(df)
        df = df.dropna(subset=['local_img_path'])
        print(f"[DataLoader] Dropped {initial_len - len(df)} rows due to image download failures.")

        if df.empty:
            raise ValueError("No data left after downloading images.")

        # 4. Extract Peripheral Features
        df = add_peripheral(df, text_col=self.text_col, path_col='local_img_path')
        
        # 5. Extract Image Central (VGG16)
        df = self.vgg.run(df, path_col='local_img_path', output_col='img_central')
        
        # 6. Extract Text Central (BERT)
        print("[DataLoader] Extracting BERT features...")
        texts = df[self.text_col].fillna("").astype(str).tolist()
        bert_tensor = self.bert.encode_texts(texts)
        df['text_central'] = list(bert_tensor.numpy())

        # 7. Split and Save
        self._split_and_save(df)
        print("[DataLoader] Processing complete.")

    def _split_and_save(self, df):
        train_full, test = train_test_split(df, test_size=self.test_size, random_state=42)
        val_ratio = self.val_size / (1.0 - self.test_size)
        train, val = train_test_split(train_full, test_size=val_ratio, random_state=42)
        
        save_parquet(train, self.train_path)
        save_parquet(val, self.val_path)
        save_parquet(test, self.test_path)