# src/data.py
import os
from typing import Optional, Tuple
import gzip
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.path import RAW_PATH, PROCESSED_PATH
from src.utils import save_parquet

from .vgg16 import VGG16Extractor
from .peripheral_features import add_text_and_image_peripheral
from .bert import BertExtractor

# 네가 앞에서 만든/만들 예정인 모듈들
# - BertExtractor: 텍스트 -> 768 (CLS) 임베딩
# from src.bert import BertExtractor

# # - (선택) VGG16Extractor: large_image_url -> 4096
# #   이미 vgg16 모듈화 해두었다면 이걸 쓰면 됨.
# try:
#     from src.vgg16 import VGG16Extractor
# except Exception:
#     VGG16Extractor = None

# # - (선택) peripheral: text_peripheral(4), image_peripheral(4)
# try:
#     from src.peripheral features import add_text_and_image_peripheral
# except Exception:
#     add_text_and_image_peripheral = None


class DataLoader:
    """
    목표: model/proposed.py (멀티모달 RHP) 학습에 필요한 컬럼을 보장
      - bert (768)
      - vgg16 (4096)
      - text_peripheral (4)
      - image_peripheral (4)
      - log_vote (label)
    """

    def __init__(
        self,
        fname: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        # raw_ext: str = "gz",
        # raw 컬럼명(필요 시 config에서 바꿀 수 있게)
        text_col: str = "clean_review",
        url_col: str = "large_image_url",
        label_col: str = "helpful_vote",
        # feature 생성 옵션
        make_bert: bool = True,
        make_vgg16: bool = True,
        make_peripheral: bool = True,
    ):
        self.fname = fname
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state


        self.text_col = text_col
        self.url_col = url_col
        self.label_col = label_col

        self.make_bert = make_bert
        self.make_vgg16 = make_vgg16
        self.make_peripheral = make_peripheral

        if not (0.0 < self.val_size < self.test_size < 1.0):
            raise ValueError(
                f"`val_size`({self.val_size}) < `test_size`({self.test_size}) < 1.0 이어야 합니다."
            )

        # 1) raw 로드
        df_raw = self._load_raw()

        # 3) (선택) features 생성: bert/vgg16/peripheral
        df_feat = self._make_features(df_raw)

        # 4) train/val/test split
        self.train, self.val, self.test = self._data_split(df_feat)

        # 5) parquet 저장
        self._save_processed()

    # ------------------------------------------------------------------
    # 1) raw 로드
    # ------------------------------------------------------------------
    def _load_raw(self) -> pd.DataFrame:
        raw_path = os.path.join(RAW_PATH, f"{self.fname}.json.gz")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw file not found: {raw_path}")

        records = []
        # gzip 안에서 줄 단위로 JSON 읽기
        with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 빈 줄 무시
                    records.append(json.loads(line))

        df = pd.DataFrame(records)
        print(f"[DataLoader] Raw loaded: {df.shape}")
        return df

    # ------------------------------------------------------------------
    # 3) feature 생성 / 정리
    # ------------------------------------------------------------------
    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ===============================
        # 0) label 처리 (helpful_vote → log_vote)
        # ===============================
        if self.label_col not in df.columns:
            raise KeyError(
                f"Label column `{self.label_col}` not found. "
                f"Available columns: {list(df.columns)}"
            )

        if self.label_col != "log_vote":
            df["log_vote"] = np.log1p(df[self.label_col])
        else:
            df["log_vote"] = df[self.label_col]
        
        # ===============================
        # 1) BERT (없을 때만 생성)
        # ===============================
        if self.make_bert:
            if "bert" not in df.columns:
                if self.text_col not in df.columns:
                    raise KeyError(f"Text column `{self.text_col}` not found.")
                extractor = BertExtractor(
                    model_ckpt="bert-base-uncased",
                    batch_size=8,
                    max_length=512,
                    use_gpu=True,
                )
                df = extractor.run(df, text_col=self.text_col, output_col="bert")
                print(f"[DataLoader] BERT generated: {df.shape}")
            else:
                print("[DataLoader] BERT exists → reuse")
                
        def get_first_large_image(x):
            if isinstance(x, list) and len(x) > 0:
                return x[0].get("large_image_url", None)
            return None

        if "review_images" in df.columns:
            df["large_image_url"] = df["review_images"].apply(get_first_large_image)

        # ===============================
        # 2) VGG16 (없을 때만 생성)
        # ===============================
        if self.make_vgg16:
            if "vgg16" not in df.columns:
                if VGG16Extractor is None:
                    raise ImportError("VGG16Extractor not available.")
                if self.url_col not in df.columns:
                    raise KeyError(f"URL column `{self.url_col}` not found.")

                vgg = VGG16Extractor(use_gpu=True)
                df = vgg.run(df, url_col=self.url_col, output_col="vgg16")
                print(f"[DataLoader] VGG16 generated: {df.shape}")
            else:
                print("[DataLoader] VGG16 exists → reuse")

        # ===============================
        # 3) Peripheral (없을 때만 생성)
        # ===============================
        if self.make_peripheral:
            missing = [c for c in ["text_peripheral", "image_peripheral"] if c not in df.columns]
            if missing:
                if add_text_and_image_peripheral is None:
                    raise ImportError("Peripheral extractor not available.")

                df = add_text_and_image_peripheral(
                    df,
                    text_col=self.text_col,
                    url_col=self.url_col,
                    text_out="text_peripheral",
                    img_out="image_peripheral",
                )
                print(f"[DataLoader] Peripheral generated: {df.shape}")
            else:
                print("[DataLoader] Peripheral exists → reuse")

       # ===============================
        # 4) 필수 입력 컬럼 검증 (모델 기준)
        # ===============================
        required_cols = [
            "bert",
            "vgg16",
            "text_peripheral",
            "image_peripheral",
            "log_vote",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Required columns missing: {missing}")


        # ===============================
        # 5) 결측 제거
        # ===============================
        df = self._drop_invalid_rows(df)

        print(f"[DataLoader] Final features ready: {df.shape}")
        return df

    def _drop_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        네 원본에서 하던 것처럼,
        - 텐서/리스트가 깨졌거나
        - peripheral이 [nan,nan,nan,nan] 같은 케이스
        를 최대한 제거.
        """
        df = df.copy()

        # bert/vgg16가 NaN이면 제거
        df = df.dropna(subset=["bert", "vgg16", "text_peripheral", "image_peripheral", "log_vote"])

        # peripheral이 "전부 nan"인 리스트면 제거
        def all_nan_list(x):
            if isinstance(x, list) and len(x) > 0:
                return all(pd.isna(v) for v in x)
            return False

        df = df[~df["text_peripheral"].apply(all_nan_list)]
        df = df[~df["image_peripheral"].apply(all_nan_list)]

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 4) train/val/test split
    # ------------------------------------------------------------------
    def _data_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.copy()

        train_full, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        val_ratio_within_train = self.val_size / (1.0 - self.test_size)

        train, val = train_test_split(
            train_full,
            test_size=val_ratio_within_train,
            random_state=self.random_state,
        )

        print(f"[DataLoader] Train: {train.shape}")
        print(f"[DataLoader] Val:   {val.shape}")
        print(f"[DataLoader] Test:  {test.shape}")

        return train, val, test

    # ------------------------------------------------------------------
    # 5) parquet 저장
    # ------------------------------------------------------------------
    def _save_processed(self):
        os.makedirs(PROCESSED_PATH, exist_ok=True)

        train_path = os.path.join(PROCESSED_PATH, f"{self.fname}_train.parquet")
        val_path = os.path.join(PROCESSED_PATH, f"{self.fname}_val.parquet")
        test_path = os.path.join(PROCESSED_PATH, f"{self.fname}_test.parquet")

        save_parquet(self.train, train_path)
        save_parquet(self.val, val_path)
        save_parquet(self.test, test_path)

        print(f"[DataLoader] Saved train/val/test parquet to {PROCESSED_PATH}")
