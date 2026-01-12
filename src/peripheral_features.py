# src/feature/peripheral.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import concurrent.futures

import cv2
from textblob import TextBlob
import textstat
from tqdm import tqdm

tqdm.pandas()


# ============================================================
# 1) Text Peripheral (sentiment, subjectivity, readability, extremity)
# ============================================================

def analyze_text_peripheral(text: str) -> List[float]:
    """
    Returns: [polarity, subjectivity, readability, extremity]
    """
    if not isinstance(text, str) or not text.strip():
        return [np.nan, np.nan, np.nan, np.nan]

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    try:
        readability = textstat.flesch_reading_ease(text)
    except Exception:
        readability = np.nan

    extremity = abs(polarity)
    return [float(polarity), float(subjectivity), float(readability), float(extremity)]


def add_text_peripheral_features(
    df: pd.DataFrame,
    text_col: str = "text",
    output_col: str = "text_peripheral",
) -> pd.DataFrame:
    df = df.copy()
    df[output_col] = df[text_col].progress_apply(analyze_text_peripheral)
    return df


# ============================================================
# 2) Image Peripheral (brightness, contrast, saturation, edge_intensity)
# ============================================================

def _parse_urls(cell: Union[str, List[str]]) -> List[str]:
    """
    Robust URL parser:
    - list[str] 그대로 사용
    - "url1', 'url2" 형태 / 콤마 구분 / 단일 url 모두 대응
    """
    if isinstance(cell, list):
        return [str(u).strip() for u in cell if isinstance(u, str) and u.strip()]

    if not isinstance(cell, str):
        return []

    s = cell.strip()
    if not s:
        return []

    if "', '" in s:
        parts = [p.strip(" '\"") for p in s.split("', '")]
        return [p for p in parts if p]
    if "," in s and s.count("http") >= 2:
        parts = [p.strip(" '\"") for p in s.split(",")]
        return [p for p in parts if p]

    return [s.strip(" '\"")]


def _download_image_cv2(url: str, timeout: int = 5, headers: Optional[dict] = None) -> Optional[np.ndarray]:
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        arr = np.asarray(bytearray(r.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None


def _image_stats(img_bgr: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns: brightness, contrast, avg_saturation, edge_intensity
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(gray.std())

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    avg_saturation = float(np.mean(saturation))

    edges = cv2.Canny(gray, 100, 200)
    edge_intensity = float(np.mean(edges))

    return brightness, contrast, avg_saturation, edge_intensity


def process_single_image(
    url: str,
    timeout: int = 5,
    headers: Optional[dict] = None,
) -> Optional[Tuple[float, float, float, float]]:
    img = _download_image_cv2(url, timeout=timeout, headers=headers)
    if img is None:
        return None
    return _image_stats(img)


def process_images_peripheral(
    urls_cell: Union[str, List[str]],
    timeout: int = 5,
    max_workers: int = 16,
    max_images_per_row: Optional[int] = None,
    headers: Optional[dict] = None,
) -> List[float]:
    """
    여러 이미지가 있으면 평균으로 aggregate.
    Returns: [avg_brightness, avg_contrast, avg_saturation, avg_edge_intensity]
    """
    urls = _parse_urls(urls_cell)
    if not urls:
        return [np.nan, np.nan, np.nan, np.nan]

    if max_images_per_row is not None:
        urls = urls[:max_images_per_row]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda u: process_single_image(u, timeout=timeout, headers=headers), urls))

    results = [r for r in results if r is not None]
    if not results:
        return [np.nan, np.nan, np.nan, np.nan]

    arr = np.array(results, dtype="float32")  # (N, 4)
    avg = np.mean(arr, axis=0)
    return avg.tolist()


def add_image_peripheral_features(
    df: pd.DataFrame,
    url_col: str = "large_image_url",
    output_col: str = "image_peripheral",
    timeout: int = 5,
    max_workers: int = 16,
    max_images_per_row: Optional[int] = None,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    df = df.copy()
    headers = {"User-Agent": user_agent}

    # progress bar를 위해 tqdm 사용
    outputs = []
    for cell in tqdm(df[url_col].tolist(), desc="Image Peripheral"):
        outputs.append(
            process_images_peripheral(
                cell,
                timeout=timeout,
                max_workers=max_workers,
                max_images_per_row=max_images_per_row,
                headers=headers,
            )
        )
    df[output_col] = outputs
    return df


# ============================================================
# 3) Convenience pipeline
# ============================================================

def add_text_and_image_peripheral(
    df: pd.DataFrame,
    text_col: str = "text",
    url_col: str = "large_image_url",
    text_out: str = "text_peripheral",
    img_out: str = "image_peripheral",
    timeout: int = 5,
    max_workers: int = 16,
    max_images_per_row: Optional[int] = None,
) -> pd.DataFrame:
    """
    원본 코드에서 하던 것처럼 두 컬럼을 한 번에 생성.
    """
    df = add_text_peripheral_features(df, text_col=text_col, output_col=text_out)
    df = add_image_peripheral_features(
        df,
        url_col=url_col,
        output_col=img_out,
        timeout=timeout,
        max_workers=max_workers,
        max_images_per_row=max_images_per_row,
    )
    return df
