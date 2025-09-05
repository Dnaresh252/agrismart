# AgriSmart

Crop analysis using CNN + LSTM — Django backend and model pipeline

---

## Overview

AgriSmart provides a backend and model pipeline that combines a convolutional neural network (CNN) for spatial feature extraction with a long short‑term memory network (LSTM) for temporal modeling. The repository includes the Django application, training scripts, example data, and saved model artifacts for running inference and retraining.

---

## Features

* CNN for image feature extraction.
* LSTM for sequence/time‑series modeling (sensor data, sequences of image features, etc.).
* Training and inference scripts with example data.
* Django app exposing prediction routes and test pages.

---

## Repository structure

```
agrismart/
├─ CropApp/           # Django app (views, templates, urls)
├─ dataset/           # Example data for training and testing
├─ model/             # Trained models and training scripts
├─ testdata.csv       # Sample input for quick tests
├─ requirements.txt   # Python dependencies
├─ manage.py          # Django entry point
├─ Procfile           # Deployment config
├─ build.sh           # Build helper script
```

---

## Prerequisites

* Python 3.8 or later
* Virtual environment recommended
* GPU optional for faster training

---

## Quick start

1. Clone the repository

```bash
git clone https://github.com/Dnaresh252/agrismart.git
cd agrismart
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the Django server

```bash
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

---

## Inference example

Adjust to match model and preprocessing steps used during training.

```python
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = 'model/cnn_lstm.h5'
model = load_model(MODEL_PATH)

# Example input (replace with real preprocessed data)
X = np.zeros((1, 10, 128))
pred = model.predict(X)
print(pred)
```

---

## Training workflow

1. Train CNN on labeled images.
2. Save features or the full CNN model.
3. Create sequences of features.
4. Train LSTM on those sequences.

Example (if training scripts are present):

```bash
python model/train_cnn.py
python model/train_lstm.py
```

---

## Common issues

* **Shape mismatch**: check input shapes with `model.summary()`.
* **Preprocessing differences**: use the same normalization and sequence length as in training.
* **Slow training**: lower batch size or simplify model if GPU is not available.

---

## Contributing

Contributions are welcome. Suggested steps:

1. Fork this repository.
2. Create a feature branch.
3. Commit changes with clear messages.
4. Open a pull request.

---

## Notes

This repository does not include a license file. All rights are reserved to the author. If you plan to use or modify the code, please contact the repository owner. The README is structured to be clear and practical, matching the project without external content.
