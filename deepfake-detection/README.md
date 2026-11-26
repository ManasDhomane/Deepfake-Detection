# Deepfake Detection using Deep Learning

EfficientNet-B0 (TensorFlow) + MTCNN face cropping, Flask API, React UI.

## One-click start
Install **Python 3.11** (3.9-3.11 work), then:

- **Windows:** double-click `start.bat`
- **Mac/Linux:** `./start.sh`

First run automatically: creates a virtual environment -> installs packages -> downloads a balanced face dataset
from Hugging Face -> trains the model -> evaluates it -> opens the app at http://localhost:5000.
Later runs skip straight to the app. Needs internet on the first run.

Options (add after the command): `--per-class 5000` more data / better accuracy (default 3000 on CPU, 8000 with an NVIDIA GPU on Linux/WSL),
`--retrain`, `--redownload`. Epochs: set `DF_EPOCHS_HEAD` / `DF_EPOCHS_FT` env vars.

Outputs: `models/deepfake_effnetb0.keras`, `reports/metrics.json`, `confusion_matrix.png`, `roc_curve.png`, `training_curves.png`.
Put these real numbers and figures in Chapter 5 of your report.

## Using FaceForensics++ instead (as in the report)
Accept the terms at https://github.com/ondyari/FaceForensics, download the c23 videos, then:
```
.venv/bin/python ml/extract_faces.py --ff-root /path/to/FaceForensics      # Windows: .venv\Scripts\python
.venv/bin/python ml/train.py && .venv/bin/python ml/evaluate.py
```
then run `start.bat` / `./start.sh`.

## Layout
`ml/` data + training + evaluation | `backend/` Flask API + inference | `frontend/` React (pre-built in `frontend/dist`)
API: `POST /api/predict` (multipart `file`) -> `{label, confidence, fake_probability, frames_analyzed, frame_scores, kind}`
