"""Accuracy / Precision / Recall / F1 / AUC-ROC, confusion matrix and ROC curve on the held-out test split."""
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

from config import *


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "auc_roc": roc_auc_score(y, p)}


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    thr = load_threshold()
    model = tf.keras.models.load_model(MODEL_PATH)
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", class_names=CLASS_NAMES, label_mode="binary", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False)
    files = ds.file_paths
    y = np.concatenate([lab.numpy().ravel() for _, lab in ds]).astype(int)
    p = model.predict(ds).ravel()

    frame = scores(y, p, thr)
    by_video = defaultdict(list)
    for f, yy, pp in zip(files, y, p):
        by_video[f.replace("\\", "/").split("/")[-1].split("__f")[0] + f"|{yy}"].append(pp)
    vy = np.array([int(k.split("|")[1]) for k in by_video])
    vp = np.array([np.mean(v) for v in by_video.values()])   # average frame scores -> video score

    out = {"threshold": thr, "frame_level": frame, "video_level": scores(vy, vp, thr),
           "mean_fake_score_on_real_images": float(p[y == 0].mean()),
           "mean_fake_score_on_fake_images": float(p[y == 1].mean()),
           "test_frames": len(y), "test_videos": len(vy)}
    json.dump(out, open(REPORT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    ConfusionMatrixDisplay(confusion_matrix(y, (p >= thr).astype(int)), display_labels=CLASS_NAMES).plot(cmap="Blues")
    plt.title("Confusion matrix (frame level)"); plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150); plt.close()
    fpr, tpr, _ = roc_curve(y, p)
    plt.plot(fpr, tpr, label=f"AUC = {frame['auc_roc']:.3f}"); plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.legend()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=150)


if __name__ == "__main__":
    main()
