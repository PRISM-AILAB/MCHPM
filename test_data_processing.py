# test_data_processing.py

import os
import pandas as pd
from src.data import DataLoader
from src.path import PROCESSED_PATH, RAW_PATH

def test_pipeline():
    # ---------------------------------------------------------
    # 1. 配置部分 (根据你的实际文件名修改)
    # ---------------------------------------------------------
    FNAME = "All_Beauty"        # 你的文件名 (不带 .json.gz)
    RAW_EXT = "jsonl.gz"         # 扩展名
    
    # ---------------------------------------------------------
    # 2. 自动检查列名 (不用你手动改来改去)
    # ---------------------------------------------------------
    print(f">>> 正在检查 {FNAME}.{RAW_EXT} 的列名...")
    raw_file = os.path.join(RAW_PATH, f"{FNAME}.{RAW_EXT}")
    
    if not os.path.exists(raw_file):
        print(f"❌ 错误: 找不到文件 {raw_file}")
        return

    # 预读取一行来确定列名
    import gzip, json
    with gzip.open(raw_file, 'rb') as f:
        first_line = json.loads(f.readline())
        keys = first_line.keys()
        print(f"   包含列: {list(keys)}")

    # 自动选择文本列
    if 'reviewText' in keys:
        text_col = 'reviewText'
    elif 'text' in keys:
        text_col = 'text'
    elif 'body' in keys:
        text_col = 'body'
    else:
        text_col = 'clean_text' # 默认值
    
    # 自动选择图片列
    if 'large_image_url' in keys:
        img_col = 'large_image_url'
    elif 'image' in keys:
        img_col = 'image'
    elif 'images' in keys:
        img_col = 'images'
    else:
        print("⚠️ 警告: 没找到图片列，程序可能会报错")
        img_col = 'large_image_url'

    print(f"✅ 选定列名: Text='{text_col}', Image='{img_col}'")

    # ---------------------------------------------------------
    # 3. 运行 DataLoader (测试核心逻辑)
    # ---------------------------------------------------------
    print("\n>>> 开始运行数据处理流程...")
    
    try:
        loader = DataLoader(
            fname=FNAME,
            raw_ext=RAW_EXT,
            test_size=0.2,
            val_size=0.1,
            text_col=text_col,
            img_col=img_col,
            device="cuda" # 如果显存不够或报错，改成 "cpu"
        )
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        return

    # ---------------------------------------------------------
    # 4. 验证结果
    # ---------------------------------------------------------
    print("\n>>> 正在验证生成的文件...")
    train_path = os.path.join(PROCESSED_PATH, f"{FNAME}_train.parquet")
    
    if os.path.exists(train_path):
        df = pd.read_parquet(train_path)
        print(f"✅ 成功! Train 数据形状: {df.shape}")
        
        # 检查特征列
        required = ['text_central', 'img_central', 'text_peripheral', 'image_peripheral', 'local_img_path']
        missing = [c for c in required if c not in df.columns]
        
        if not missing:
            print("✅ 所有特征列均已生成!")
            print("-" * 30)
            row = df.iloc[0]
            print(f"Sample Local Path: {row['local_img_path']}")
            print(f"Text Peri (Sentiment, etc): {row['text_peripheral']}")
            print(f"Image Peri (Bright, etc):   {row['image_peripheral']}")
            print(f"BERT Shape: {len(row['text_central'])}")
            print(f"VGG Shape:  {len(row['img_central'])}")
            print("-" * 30)
        else:
            print(f"❌ 缺少特征列: {missing}")
    else:
        print("❌ 未找到生成的 Parquet 文件。")

if __name__ == "__main__":
    test_pipeline()