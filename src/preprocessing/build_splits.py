from pathlib import Path

import pandas as pd

from config import FEATURE_COLUMNS, LABEL_COLUMN, LABEL_MAP
from split_config import RAW_DATA_DIR


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def split_file(file_path):
    columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    df = pd.read_parquet(
        file_path,
        columns=columns,
    )

    df["attack_family"] = df[LABEL_COLUMN].map(LABEL_MAP)

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["attack_family"]
    )

    total = len(df)

    train_end = int(total * TRAIN_RATIO)
    validation_end = int(
        total * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def main():
    print("=" * 70)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 70)

    files = sorted(
        RAW_DATA_DIR.glob("*.parquet")
    )

    total_train = 0
    total_validation = 0
    total_test = 0

    for file_path in files:
        train, validation, test = split_file(
            file_path
        )

        total_train += len(train)
        total_validation += len(validation)
        total_test += len(test)

        print(
            f"\n{file_path.name}"
        )
        print(
            f"  Train:      {len(train):,}"
        )
        print(
            f"  Validation: {len(validation):,}"
        )
        print(
            f"  Test:       {len(test):,}"
        )

    print("\n" + "=" * 70)
    print("TOTALS")
    print("=" * 70)

    print(f"Train:      {total_train:,}")
    print(f"Validation: {total_validation:,}")
    print(f"Test:       {total_test:,}")

    total = (
        total_train
        + total_validation
        + total_test
    )

    print(f"Combined:   {total:,}")


if __name__ == "__main__":
    main()