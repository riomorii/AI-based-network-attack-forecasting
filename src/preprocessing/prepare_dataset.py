import numpy as np
import pandas as pd

from pathlib import Path

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_FILE,
    LABEL_COLUMN,
    FEATURE_COLUMNS,
    LABEL_MAP,
    LABEL_TO_ID,
)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean numeric network-flow features.
    """

    # Convert selected features to numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace positive/negative infinity with NaN
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows containing invalid feature values
    df = df.dropna(subset=FEATURE_COLUMNS)

    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw CIC-IDS2018 labels into attack families.
    """

    df["attack_family"] = df[LABEL_COLUMN].map(LABEL_MAP)

    # Keep only labels we explicitly understand
    df = df.dropna(subset=["attack_family"])

    # Convert attack family to numeric class ID
    df["label_id"] = df["attack_family"].map(LABEL_TO_ID)

    return df


def process_file(file_path: Path) -> pd.DataFrame:
    """
    Process one Parquet file at a time.
    """

    print(f"\nProcessing: {file_path.name}")

    columns_to_read = FEATURE_COLUMNS + [LABEL_COLUMN]

    df = pd.read_parquet(
        file_path,
        columns=columns_to_read,
    )

    original_rows = len(df)

    df = clean_features(df)
    after_cleaning = len(df)

    df = map_labels(df)
    after_labels = len(df)

    result = df[
        FEATURE_COLUMNS
        + ["attack_family", "label_id"]
    ].copy()

    print(f"  Original rows: {original_rows:,}")
    print(f"  After cleaning: {after_cleaning:,}")
    print(f"  After label mapping: {after_labels:,}")

    return result


def main():
    PROCESSED_DATA_DIR.mkdir(
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

    print("=" * 70)
    print("CSE-CIC-IDS2018 PREPROCESSING")
    print("=" * 70)

    processed_parts = []

    for file_path in files:
        processed = process_file(file_path)
        processed_parts.append(processed)

    print("\nCombining processed files...")

    final_df = pd.concat(
        processed_parts,
        ignore_index=True,
    )

    print(f"Final rows: {len(final_df):,}")

    print("\nClass distribution:")
    print(
        final_df["attack_family"]
        .value_counts()
        .to_string()
    )

    print("\nSaving processed dataset...")

    final_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print("\nDONE.")


if __name__ == "__main__":
    main()