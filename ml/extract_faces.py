"""Steps 1-6 of the report: frames -> MTCNN face crops -> 224x224 -> video-wise 70/15/15 split.

Expected FaceForensics++ layout (download via the official script after accepting the terms):
  <ff_root>/original_sequences/youtube/c23/videos/*.mp4
  <ff_root>/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/*.mp4

Usage:
  python ml/extract_faces.py --ff-root /path/to/FaceForensics
"""
import argparse, random
from pathlib import Path

import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

from config import DATA_DIR, IMG_SIZE, SEED

FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def collect_videos(root: Path, comp: str):
    items = [(v, "real") for v in sorted((root / "original_sequences/youtube" / comp / "videos").glob("*.mp4"))]
    for m in FAKE_METHODS:
        items += [(v, "fake") for v in sorted((root / "manipulated_sequences" / m / comp / "videos").glob("*.mp4"))]
    return items


def split_by_identity(videos):
    """Fake '000_003.mp4' reuses the target video '000', so group by the first id
    to stop the same face appearing in both train and test (data leakage)."""
    ids = sorted({v.stem.split("_")[0] for v, _ in videos})
    random.Random(SEED).shuffle(ids)
    n = len(ids)
    train, val = set(ids[: int(.70 * n)]), set(ids[int(.70 * n): int(.85 * n)])
    return lambda vid: "train" if vid in train else "val" if vid in val else "test"


def crop_face(detector, bgr, margin=0.2):
    faces = detector.detect_faces(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    faces = [f for f in faces if f["confidence"] > 0.9]
    if not faces:
        return None
    x, y, w, h = max(faces, key=lambda f: f["confidence"])["box"]
    x, y = max(x, 0), max(y, 0)
    dx, dy = int(w * margin), int(h * margin)
    H, W = bgr.shape[:2]
    face = bgr[max(y - dy, 0): min(y + h + dy, H), max(x - dx, 0): min(x + w + dx, W)]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) if face.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ff-root", required=True, type=Path)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--real-frames", type=int, default=40, help="frames per real video")
    ap.add_argument("--fake-frames", type=int, default=10, help="frames per fake video (4 methods -> balances classes)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N videos")
    a = ap.parse_args()

    videos = collect_videos(a.ff_root, a.compression)
    if a.limit:
        videos = videos[: a.limit]
    split_of = split_by_identity(videos)
    detector = MTCNN()

    for path, label in tqdm(videos, desc="videos"):
        split = split_of(path.stem.split("_")[0])
        out = DATA_DIR / split / label
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = a.real_frames if label == "real" else a.fake_frames
        for idx in np.linspace(0, max(total - 1, 0), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            face = crop_face(detector, frame)
            if face is not None:
                # "<video>__f<idx>.jpg" lets evaluate.py aggregate frames back to a video
                cv2.imwrite(str(out / f"{path.stem}__f{idx:05d}.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cap.release()
    print("Done ->", DATA_DIR)


if __name__ == "__main__":
    main()
