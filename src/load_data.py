import pandas as pd
import os
from pathlib import Path

# Get the project root directory (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "campaign.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"

os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)

# IMPORTANT: sep=";" for Bank Marketing dataset
df = pd.read_csv(INPUT_PATH, sep=";")

df.to_csv(OUTPUT_PATH, index=False)

print("Data loaded successfully with correct separator")
