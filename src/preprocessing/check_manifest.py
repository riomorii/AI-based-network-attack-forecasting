from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")

files = sorted(RAW_DATA_DIR.glob("*.parquet"))

print("SOURCE DATASET MANIFEST")
print("=" * 80)

for i, file_path in enumerate(files, start=1):
    df = pd.read_parquet(
        file_path,
        columns=["Label"]
    )

    print(
        f"{i:02}. {file_path.name} | "
        f"{len(df):,} flows"
    )

print("=" * 80)
print(f"TOTAL FILES: {len(files)}")