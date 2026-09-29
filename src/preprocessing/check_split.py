from pathlib import Path

import pandas as pd

from config import LABEL_MAP
from split_config import (
    RAW_DATA_DIR,
    TRAIN_FILES,
    VALIDATION_FILES,
    TEST_FILES,
)


def get_distribution(file_names):
    frames = []

    for file_name in file_names:
        file_path = RAW_DATA_DIR / file_name

        df = pd.read_parquet(
            file_path,
            columns=["Label"],
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined["attack_family"] = combined["Label"].map(
        LABEL_MAP
    )

    return combined["attack_family"].value_counts()


def main():
    splits = {
        "TRAIN": TRAIN_FILES,
        "VALIDATION": VALIDATION_FILES,
        "TEST": TEST_FILES,
    }

    print("=" * 70)
    print("SPLIT CLASS COVERAGE")
    print("=" * 70)

    for split_name, files in splits.items():
        distribution = get_distribution(files)

        print(f"\n{split_name}")
        print("-" * 40)
        print(distribution.to_string())

        print(
            f"\nTotal: {distribution.sum():,} flows"
        )

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()