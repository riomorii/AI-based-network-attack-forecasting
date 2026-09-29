import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from src.preprocessing.config import FEATURE_COLUMNS, LABEL_COLUMN
from src.sequences.sequence_config import SEQUENCE_LENGTH

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences"

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_scaler.pkl"
)


def build_sequences_from_file(file_path, scaler):
    print(f"\nProcessing: {file_path.name}")

    columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    df = pd.read_parquet(
        file_path,
        columns=columns,
    )

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [LABEL_COLUMN]
    ).reset_index(drop=True)

    features = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    labels = df[LABEL_COLUMN].to_numpy()

    features = scaler.transform(features).astype(
        np.float32
    )

    X = []
    y = []

    for i in range(
        len(features) - SEQUENCE_LENGTH
    ):
        X.append(
            features[
                i:i + SEQUENCE_LENGTH
            ]
        )

        y.append(
            labels[
                i + SEQUENCE_LENGTH
            ]
        )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    y = np.asarray(y)

    print(f"  Flows:    {len(features):,}")
    print(f"  Windows:  {len(X):,}")
    print(f"  X shape:  {X.shape}")
    print(f"  y shape:  {y.shape}")

    return X, y


def main():
    print("=" * 70)
    print("BUILDING FLOW SEQUENCES")
    print("=" * 70)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        RAW_DATA_DIR.glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {RAW_DATA_DIR}"
        )

    for file_path in files:
        X, y = build_sequences_from_file(
            file_path,
            scaler,
        )

        output_name = (
            file_path.stem
            + "_sequences.npz"
        )

        output_path = (
            OUTPUT_DIR / output_name
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
        )

        print(f"  Saved: {output_path.name}")

    print("\n" + "=" * 70)
    print("SEQUENCE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()