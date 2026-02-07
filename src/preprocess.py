import pandas as pd
from pathlib import Path

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "data.csv"

df = pd.read_csv(INPUT_PATH)

# encode everything (quick approach)
df = pd.get_dummies(df, drop_first=True)

df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing done")
