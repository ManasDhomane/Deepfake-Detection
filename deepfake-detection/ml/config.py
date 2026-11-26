import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
DATA_DIR = ROOT / "data" / "faces"          # output of download_dataset.py / extract_faces.py
MODEL_PATH = ROOT / "models" / "deepfake_effnetb0.keras"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"   # decision threshold tuned on the validation split
REPORT_DIR = ROOT / "reports"
CLASS_NAMES = ["real", "fake"]               # fake = 1, real = 0
EPOCHS_HEAD = int(os.environ.get("DF_EPOCHS_HEAD", 3))   # phase 1: train classifier head
EPOCHS_FT = int(os.environ.get("DF_EPOCHS_FT", 12))      # phase 2: fine-tune the whole network (early stopping applies)


def load_threshold():
    try:
        return float(json.load(open(THRESHOLD_PATH))["threshold"])
    except Exception:
        return 0.5
