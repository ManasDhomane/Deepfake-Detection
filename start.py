#!/usr/bin/env python3
"""One-click launcher: virtual env -> dependencies -> dataset -> training -> evaluation -> web app.
Every stage is skipped if its output already exists, so the next launch goes straight to the app.

  python start.py                   # everything
  python start.py --per-class 5000  # more training data (better accuracy, slower)
  python start.py --retrain         # train again from scratch
"""
import argparse, os, shutil, subprocess, sys, threading, time, urllib.request, venv, webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DATA, MODEL, METRICS = ROOT / "data/faces", ROOT / "models/deepfake_effnetb0.keras", ROOT / "reports/metrics.json"
URL = "http://localhost:5000"


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)


def bootstrap(args):
    if not (3, 9) <= sys.version_info[:2] <= (3, 11):
        sys.exit(f"Python 3.9, 3.10 or 3.11 is required (you have {sys.version.split()[0]}). "
                 "Install Python 3.11 from python.org and run again.")
    if not PY.exists():
        step("Creating virtual environment")
        venv.create(VENV, with_pip=True)
    stamp, req = VENV / ".deps_installed", ROOT / "requirements.txt"
    if not stamp.exists() or stamp.stat().st_mtime < req.stat().st_mtime:
        step("Installing packages (first run only, can take 5-10 minutes)")
        run([PY, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        run([PY, "-m", "pip", "install", "-r", req], check=True)
        stamp.touch()
    sys.exit(run([PY, __file__, "--in-venv", *sys.argv[1:]]).returncode)


def has_images(p):
    return p.exists() and any(p.rglob("*.jpg"))


def serve():
    step(f"Starting the app at {URL}  (press Ctrl+C to stop)")
    server = subprocess.Popen([str(PY), str(ROOT / "backend/app.py")], cwd=ROOT)

    def open_when_ready():
        for _ in range(120):
            try:
                urllib.request.urlopen(URL + "/api/health", timeout=1)
                webbrowser.open(URL)
                return
            except Exception:
                time.sleep(1)
    threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-venv", action="store_true")
    ap.add_argument("--per-class", type=int, default=0, help="training images per class (default: 3000 CPU / 8000 GPU)")
    ap.add_argument("--redownload", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    if not a.in_venv:
        bootstrap(a)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    per_class = a.per_class or (8000 if shutil.which("nvidia-smi") and os.name != "nt" else 3000)  # native Windows TensorFlow has no GPU support

    if a.redownload or not has_images(DATA / "train"):
        step(f"Downloading dataset ({per_class} images per class for training)")
        if run([PY, "ml/download_dataset.py", "--per-class", per_class]).returncode:
            sys.exit("\nDataset download failed. Check your internet connection and run again.\n"
                     "Alternative: use FaceForensics++ with  python ml/extract_faces.py --ff-root <path>  (see README).")
        a.retrain = True

    if a.retrain or not MODEL.exists():
        step("Training the EfficientNet-B0 model (slow on CPU - this is the longest step, runs once)")
        if run([PY, "ml/train.py"]).returncode:
            sys.exit("Training failed - see the error above.")
        METRICS.unlink(missing_ok=True)

    if not METRICS.exists():
        step("Evaluating on the test set")
        run([PY, "ml/evaluate.py"])

    serve()


if __name__ == "__main__":
    main()
