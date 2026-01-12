# main.py
import os
import tensorflow as tf
from tensorflow import keras

from src.path import PROCESSED_PATH, SAVE_MODEL_PATH, UTILS_PATH
from src.utils import load_yaml, load_parquet, set_seed, get_metrics
from src.data import DataLoader
from model.proposed import RHP, get_data_loader, rhp_trainer, rhp_tester, CoAttentionBlock, GateComplement


if __name__ == "__main__":
    # 1) config 로드
    CONFIG_FPATH = os.path.join(UTILS_PATH, "config.yaml")
    cfg = load_yaml(CONFIG_FPATH)

    dargs = cfg.get("data", {})
    args = cfg.get("args", {})

    FNAME = dargs.get("fname")
    args["fname"] = FNAME

    set_seed(cfg.get("seed", 42))

    # 2) processed train/val/test parquet 존재 여부 확인
    train_path = os.path.join(PROCESSED_PATH, f"{FNAME}_train.parquet")
    val_path   = os.path.join(PROCESSED_PATH, f"{FNAME}_val.parquet")
    test_path  = os.path.join(PROCESSED_PATH, f"{FNAME}_test.parquet")

    need_processing = not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path))

    if need_processing:
        print("[main] Processed parquet not found. Running DataLoader from raw...")
        loader = DataLoader(
            fname=FNAME,
            test_size=dargs.get("test_size", 0.2),
            val_size=dargs.get("val_size", 0.1),
            random_state=cfg.get("seed", 42),
            # 🔥 이 두 줄이 핵심
            label_col=dargs.get("label_col", "helpful_vote")
        )
        # DataLoader 내부에서 parquet 저장까지 하도록 설계(참조코드 방식)
        train_df, val_df, test_df = loader.train, loader.val, loader.test
    else:
        print("[main] Found processed parquet. Loading...")
        train_df = load_parquet(train_path)
        val_df   = load_parquet(val_path)
        test_df  = load_parquet(test_path)

    print("[main] train:", train_df.shape)
    print("[main] val  :", val_df.shape)
    print("[main] test :", test_df.shape)

    # 3) tf.data.Dataset 생성
    train_loader = get_data_loader(args, train_df, shuffle=True)
    val_loader   = get_data_loader(args, val_df, shuffle=False)
    test_loader  = get_data_loader(args, test_df, shuffle=False)

    # 4) 모델 생성 (네 proposed.py 시그니처에 맞게)
    num_heads = int(args.get("num_heads", 10))
    feature_dim_per_head = int(args.get("feature_dim_per_head", 64))
    feature_dimension = feature_dim_per_head * num_heads

    model = RHP(
        feature_dimension=feature_dimension,
        num_heads=num_heads,
        dropout=float(args.get("dropout", 0.1)),
        learning_rate=float(args.get("lr", 1e-4)),
        key_dim=int(args.get("key_dim", 64)),
        dff=int(args.get("dff", 2048)),
    )
    model.summary()

    # 5) 학습
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    best_model_path = os.path.join(SAVE_MODEL_PATH, f"{FNAME}_Best_Model.keras")

    rhp_trainer(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        best_model_path=best_model_path,
    )

    # 6) Best 모델 로드 + 테스트
    best_model = keras.models.load_model(best_model_path, compile=False,
                                         custom_objects={
        "CoAttentionBlock": CoAttentionBlock,
         "GateComplement": GateComplement,
    },)
    test_preds, test_trues = rhp_tester(
        args=args,
        model=best_model,
        test_loader=test_loader,
    )

    mse, rmse, mae, mape = get_metrics(test_preds, test_trues)
    print(f"[TEST] RMSE={rmse:.5f}  MSE={mse:.5f}  MAE={mae:.5f}  MAPE={mape:.3f}%")
