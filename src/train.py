import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

df = pd.read_csv(INPUT_PATH)

TARGET_COL = "y_yes"



X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

joblib.dump(model, MODEL_PATH)

print("Training complete")
print("Accuracy:", acc)
