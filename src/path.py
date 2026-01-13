import os

# Define Root Directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data Paths
DATA_PATH = os.path.join(ROOT_DIR, "data")
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")

# Source Code / Config Path
UTILS_PATH = os.path.join(ROOT_DIR, "src")

# Model Paths
MODEL_PATH = os.path.join(ROOT_DIR, "model")
SAVE_MODEL_PATH = os.path.join(MODEL_PATH, "save")

# Create directories if they don't exist
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)