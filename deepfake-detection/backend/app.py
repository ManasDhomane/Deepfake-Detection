import os, tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from predictor import NoFaceFound, Predictor

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"   # pre-built React app

app = Flask(__name__, static_folder=str(DIST), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB
CORS(app)
_predictor = None


def predictor():
    global _predictor
    if _predictor is None:                              # load once, on first request
        _predictor = Predictor()
    return _predictor


@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        return "Frontend not built. Run: cd frontend && npm install && npm run build", 503
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file uploaded."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        return jsonify(error=f"Unsupported file type {ext or '(none)'}. Use JPG, PNG, WEBP, MP4, MOV, AVI, MKV or WEBM."), 415

    fd, tmp = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(tmp)
        p = predictor()
        return jsonify(p.predict_image(tmp) if ext in IMAGE_EXT else p.predict_video(tmp))
    except NoFaceFound:
        return jsonify(error="No clear face was found. Upload a file where a face is visible and well lit."), 422
    except ValueError as e:
        return jsonify(error=str(e)), 400
    finally:
        os.remove(tmp)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File is larger than 100 MB."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
