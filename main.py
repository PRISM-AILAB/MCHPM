import os
import torch
import numpy as np
import pandas as pd

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH, UTILS_PATH
from src.utils import load_yaml, load_parquet, set_seed, get_metrics
from src.data import DataLoader
from model.proposed import RHP, get_data_loader, rhp_trainer, rhp_tester

def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    config_path = os.path.join(UTILS_PATH, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    cfg = load_yaml(config_path)
    dargs = cfg.get("data", {})
    args = cfg.get("args", {})
    
    fname = dargs.get("fname", "All_Beauty")
    args["fname"] = fname

    set_seed(cfg.get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    # Initialize DataLoader (automatically checks if processed data exists)
    print(f"[Main] Checking data pipeline for '{fname}'...")
    data_processor = DataLoader(
        fname=fname,
        test_size=dargs.get("test_size", 0.2),
        val_size=dargs.get("val_size", 0.1),
        text_col=dargs.get("text_col", "clean_text"),
        img_col=dargs.get("img_col", "large_image_url"),
        vote_col=dargs.get("vote_col", "vote"),
        device=str(device)
    )
    
    train_path = os.path.join(PROCESSED_PATH, f"{fname}_train.parquet")
    val_path   = os.path.join(PROCESSED_PATH, f"{fname}_val.parquet")
    test_path  = os.path.join(PROCESSED_PATH, f"{fname}_test.parquet")

    print("[Main] Loading parquet files...")
    train_df = load_parquet(train_path)
    val_df   = load_parquet(val_path)
    test_df  = load_parquet(test_path)

    print(f"   - Train shape: {train_df.shape}")
    print(f"   - Val shape:   {val_df.shape}")
    print(f"   - Test shape:  {test_df.shape}")

    train_loader = get_data_loader(args, train_df, shuffle=True)
    val_loader   = get_data_loader(args, val_df, shuffle=False)
    test_loader  = get_data_loader(args, test_df, shuffle=False)

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    num_heads = int(args.get("num_heads", 4))
    feature_dimension = int(args.get("feature_dimension", 256))
    
    print(f"[Main] Building RHP Model (Dim={feature_dimension}, Heads={num_heads})...")
    
    model = RHP(
        feature_dimension=feature_dimension,
        num_heads=num_heads,
        dropout=float(args.get("dropout", 0.1)),
        dff=int(args.get("dff", 2048))
    )

    # ==========================================
    # 4. Training
    # ==========================================
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    best_model_path = os.path.join(SAVE_MODEL_PATH, f"{fname}_Best_Model.pth")
    
    print("[Main] Starting training...")
    rhp_trainer(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=best_model_path,
    )

    # ==========================================
    # 5. Testing
    # ==========================================
    print("[Main] Loading best model for testing...")
    
    # Re-initialize and load weights
    best_model = RHP(
        feature_dimension=feature_dimension,
        num_heads=num_heads,
        dropout=float(args.get("dropout", 0.1)),
        dff=int(args.get("dff", 2048))
    )
    best_model.load_state_dict(torch.load(best_model_path))
    
    # Inference
    test_preds, test_trues = rhp_tester(
        args=args,
        model=best_model,
        test_loader=test_loader,
    )

    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    
    print("-" * 40)
    print(f"[TEST RESULTS]")
    print(f"   MAE:  {mae:.5f}")
    print(f"   MSE:  {mse:.5f}")
    print(f"   RMSE: {rmse:.5f}")
    print(f"   MAPE: {mape:.3f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()