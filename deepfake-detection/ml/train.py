"""Steps 7-8: EfficientNet-B0 transfer learning.
Phase 1 trains the new head; phase 2 fine-tunes the whole network at a lower learning rate.
Afterwards the decision threshold is tuned on the validation split (not the test split)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

from config import *


def load(split, train=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split, class_names=CLASS_NAMES, label_mode="binary", image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=None, shuffle=train, seed=SEED)
    if train:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05, fill_mode="reflect"),
            layers.RandomZoom((-0.2, 0.0), fill_mode="reflect"),      # uploaded photos frame the face differently
            layers.RandomBrightness(0.15, value_range=(0, 255)),
        ])

        def jpeg(x):                                                  # mild compression, only on some samples
            return tf.image.random_jpeg_quality(tf.clip_by_value(x, 0, 255) / 255.0, 75, 100) * 255.0

        def augment(x, y):
            x = aug(x, training=True)
            x = tf.cond(tf.random.uniform([]) < 0.25, lambda: jpeg(x), lambda: x)
            return tf.ensure_shape(x, [IMG_SIZE, IMG_SIZE, 3]), y
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    # Keras' EfficientNet has its own rescaling layer, so inputs stay in 0-255.
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out), base


def compile_model(model, lr):
    model.compile(tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=[
        "accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")])


def best_threshold(y, p):
    """Threshold with the best balanced accuracy, so the model is not biased towards one class."""
    best, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        pred = p >= t
        bal = (pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2
        if bal > best + 1e-9 or (abs(bal - best) < 1e-9 and abs(t - .5) < abs(best_t - .5)):
            best, best_t = bal, float(t)
    return best_t, float(best)


def main():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    train_ds, val_ds = load("train", True), load("val")

    n_real = len(list((DATA_DIR / "train/real").glob("*.jpg")))
    n_fake = len(list((DATA_DIR / "train/fake").glob("*.jpg")))
    total = n_real + n_fake
    class_weight = {0: total / (2 * n_real), 1: total / (2 * n_fake)}
    print("train frames:", n_real, "real,", n_fake, "fake | class weights:", class_weight)

    model, base = build_model()
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, class_weight=class_weight)

    base.trainable = True
    for layer in base.layers:                       # keep BatchNorm statistics frozen
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-4)
    cbs = [callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_auc", mode="max", save_best_only=True),
           callbacks.EarlyStopping("val_auc", mode="max", patience=4),
           callbacks.ReduceLROnPlateau("val_auc", mode="max", factor=0.3, patience=1, min_lr=1e-6)]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, class_weight=class_weight, callbacks=cbs)

    model = tf.keras.models.load_model(MODEL_PATH)   # best epoch by validation AUC
    y = np.concatenate([lab.numpy().ravel() for _, lab in val_ds]).astype(int)
    p = model.predict(val_ds).ravel()
    thr, bal = best_threshold(y, p)
    json.dump({"threshold": thr, "val_balanced_accuracy": bal}, open(THRESHOLD_PATH, "w"))
    print(f"Decision threshold {thr:.2f} (validation balanced accuracy {bal:.3f})")

    hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    json.dump(hist, open(REPORT_DIR / "history.json", "w"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, k in zip(ax, ["loss", "accuracy"]):
        a.plot(hist[k], label="train"); a.plot(hist["val_" + k], label="val"); a.set_title(k); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "training_curves.png", dpi=150)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
