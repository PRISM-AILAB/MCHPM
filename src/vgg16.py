# src/feature/vgg16.py
from __future__ import annotations

import time
from io import BytesIO
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms


class VGG16Extractor:
    """
    Extract VGG16 (fc2) features (4096-d) from image URLs.
    - Input: df[url_col] where each row is either:
        (1) list[str] of urls
        (2) a single string containing multiple urls (optionally)
        (3) a single url str
    - Output: df[output_col] as list[float] length 4096
    """

    def __init__(
        self,
        use_gpu: bool = True,
        timeout_sec: int = 10,
        sleep_sec: float = 0.0,
        user_agent: str = "Mozilla/5.0",
        max_images_per_row: Optional[int] = None,   # e.g., 3 if you want cap
        agg: str = "mean",  # "mean" or "first"
    ):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.timeout_sec = timeout_sec
        self.sleep_sec = sleep_sec
        self.headers = {"User-Agent": user_agent}
        self.max_images_per_row = max_images_per_row
        self.agg = agg

        # VGG16 pretrained, drop last classifier layer to get 4096
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])

        for p in model.features.parameters():
            p.requires_grad = False

        self.model = model.to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.session = requests.Session()

    def _parse_urls(self, x: Union[str, List[str]]) -> List[str]:
        # already a list
        if isinstance(x, list):
            urls = [str(u).strip() for u in x if isinstance(u, str) and u.strip()]
            return urls

        # string: try to split common patterns
        if not isinstance(x, str):
            return []

        s = x.strip()
        if not s:
            return []

        # common: "url1', 'url2', 'url3"
        if "', '" in s:
            parts = s.split("', '")
            parts = [p.strip(" '\"") for p in parts]
            urls = [p for p in parts if p]
        # common: comma-separated
        elif "," in s and s.count("http") >= 2:
            parts = [p.strip(" '\"") for p in s.split(",")]
            urls = [p for p in parts if p]
        else:
            urls = [s.strip(" '\"")]

        return urls

    def _fetch_image(self, url: str) -> Optional[Image.Image]:
        try:
            r = self.session.get(url, headers=self.headers, timeout=self.timeout_sec)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        except Exception:
            return None

    @torch.no_grad()
    def _encode_one(self, url: str) -> Optional[np.ndarray]:
        img = self._fetch_image(url)
        if img is None:
            return None

        x = self.preprocess(img).unsqueeze(0).to(self.device)  # [1,3,224,224]
        feat = self.model(x)  # [1, 4096]
        return feat.squeeze(0).detach().cpu().numpy()

    def encode_urls(self, urls: List[str]) -> Optional[np.ndarray]:
        if not urls:
            return None

        if self.max_images_per_row is not None:
            urls = urls[: self.max_images_per_row]

        feats = []
        for url in urls:
            feat = self._encode_one(url)
            if feat is not None and isinstance(feat, np.ndarray):
                feats.append(feat)

            if self.sleep_sec > 0:
                time.sleep(self.sleep_sec)

        if not feats:
            return None

        if self.agg == "first":
            return feats[0]
        # default mean
        return np.mean(np.stack(feats, axis=0), axis=0)

    def run(
        self,
        df: pd.DataFrame,
        url_col: str,
        output_col: str = "vgg16",
        verbose_every: int = 500,
    ) -> pd.DataFrame:
        if url_col not in df.columns:
            raise KeyError(f"{url_col} column not found in DataFrame.")

        df = df.copy()
        outputs: List[Union[List[float], float]] = []

        for i, x in enumerate(tqdm(df[url_col].tolist(), desc="VGG16 Feature")):
            if verbose_every and (i % verbose_every == 0):
                # tqdm랑 같이 써도 되고, 로그 원하면 이 출력 유지
                pass

            urls = self._parse_urls(x)
            feat = self.encode_urls(urls)

            if feat is None:
                outputs.append(np.nan)
            else:
                outputs.append(feat.astype("float32").tolist())

        df[output_col] = outputs
        return df
