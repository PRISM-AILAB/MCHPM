# src/utils.py

import gzip
import json
import yaml
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

# --------- JSONL Utils ---------
def parse(path):
    """Generator to parse gzipped jsonl file line by line."""
    with gzip.open(path, "rb") as g:
        for l in g:
            yield json.loads(l)

def getDF(path):
    """Convert parsed jsonl data into a pandas DataFrame."""
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")

# --------- Parquet / YAML Utils ---------
def load_parquet(fpath):
    """Load parquet file using pyarrow engine."""
    return pd.read_parquet(fpath, engine="pyarrow")

def save_parquet(df: pd.DataFrame, fpath: str):
    """Save DataFrame as parquet file."""
    df.to_parquet(fpath, engine="pyarrow")

def load_yaml(fpath: str) -> dict:
    """Load configuration from a YAML file."""
    with open(fpath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# --------- Seeding ---------
def set_seed(seed: int):
    """Fix random seeds for reproducibility (numpy, torch, random)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------- Metrics ---------
def get_metrics(preds, trues):
    """
    Calculate regression metrics.
    Returns: MSE, RMSE, MAE, MAPE
    """
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    mape = mean_absolute_percentage_error(trues, preds) * 100
    return mse, rmse, mae, mape