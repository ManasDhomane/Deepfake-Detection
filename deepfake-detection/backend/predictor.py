import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

sys.path.append(str(Path(__file__).resolve().parent.parent / "ml"))
from config import IMG_SIZE, MODEL_PATH, load_threshold  # noqa: E402


class NoFaceFound(Exception):
    pass


class Predictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.detector = MTCNN()
        self.thr = load_threshold()                    # tuned on the validation split during training

    def _face(self, rgb, margin=0.2):
        faces = [f for f in self.detector.detect_faces(rgb) if f["confidence"] > 0.9]
        if not faces:
            return None
        x, y, w, h = max(faces, key=lambda f: f["confidence"])["box"]
        x, y = max(x, 0), max(y, 0)
        dx, dy = int(w * margin), int(h * margin)
        H, W = rgb.shape[:2]
        crop = rgb[max(y - dy, 0): min(y + h + dy, H), max(x - dx, 0): min(x + w + dx, W)]
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) if crop.size else None

    def _score(self, faces):
        """Fake score per face, averaged over the face and its mirror image, rescaled so 0.5 = decision boundary."""
        batch = np.stack(faces).astype("float32")
        p = (self.model.predict(batch, verbose=0).ravel() + self.model.predict(batch[:, :, ::-1], verbose=0).ravel()) / 2
        t = self.thr
        return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))

    @staticmethod
    def _result(kind, scores):
        s = float(np.mean(scores))
        return {"kind": kind, "label": "fake" if s >= 0.5 else "real",
                "confidence": round(50 + 100 * abs(s - 0.5), 1), "fake_probability": round(s, 4),
                "frames_analyzed": len(scores), "frame_scores": [round(float(x), 3) for x in scores]}

    def predict_image(self, path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError("Could not read the image.")
        face = self._face(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if face is None:
            raise NoFaceFound()
        return self._result("image", self._score([face]))

    def predict_video(self, path, n_frames=16):
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("Could not read the video.")
        faces = []
        for idx in np.linspace(0, total - 1, n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and (f := self._face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) is not None:
                faces.append(f)
        cap.release()
        if not faces:
            raise NoFaceFound()
        return self._result("video", self._score(faces))
