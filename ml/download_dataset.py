"""Downloads a balanced subset of an open face dataset from Hugging Face (no login needed) and writes
data/faces/{train,val,test}/{real,fake}/*.jpg, ready for train.py.

Dataset: ThinothW/Deepfake-Identity-Isolated-Dataset-PreP (~143k face crops: face swaps, GAN and diffusion fakes;
identities never overlap between train/val/test). Only the amount you ask for is streamed.
"""
import argparse, os

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_DIR, IMG_SIZE, SEED

REPO = "ThinothW/Deepfake-Identity-Isolated-Dataset-PreP"
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def is_real(ds, label):
    names = getattr(ds.features.get("label") if ds.features else None, "names", None)
    return names[label].lower().startswith("real") if names else label == 1   # card: 0 = fake, 1 = real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=int(os.environ.get("DF_PER_CLASS", 1500)),
                    help="training images per class; val/test get 20%% of this (min 200)")
    a = ap.parse_args()

    for out_split, hf_split in SPLITS.items():
        quota = a.per_class if out_split == "train" else max(a.per_class // 5, 200)
        ds = load_dataset(REPO, split=hf_split, streaming=True).shuffle(seed=SEED, buffer_size=10000)
        for c in ("real", "fake"):
            (DATA_DIR / out_split / c).mkdir(parents=True, exist_ok=True)
        counts, scanned = {"real": 0, "fake": 0}, 0
        bar = tqdm(total=quota * 2, desc=f"{out_split}")
        for ex in ds:
            scanned += 1
            name = "real" if is_real(ds, ex["label"]) else "fake"
            if counts[name] < quota:
                img = ex["image"].convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                img.save(DATA_DIR / out_split / name / f"img{counts[name]:06d}.jpg", quality=95)
                counts[name] += 1
                bar.update(1)
            if min(counts.values()) >= quota or scanned > quota * 80:
                break
        bar.close()
        print(out_split, counts)
    print("Done ->", DATA_DIR)


if __name__ == "__main__":
    main()
