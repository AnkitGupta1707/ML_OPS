import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics" / "metrics.json"

# Load data
df = pd.read_csv(DATA_PATH)

TARGET_COL = "y_yes"

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# Load trained model
model = joblib.load(MODEL_PATH)

# Predictions
y_pred = model.predict(X)

# Metrics
metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred),
    "f1_score": f1_score(y, y_pred)
}

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation done")
print(metrics)
