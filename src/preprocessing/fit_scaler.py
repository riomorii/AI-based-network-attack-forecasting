import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS
from split_config import RAW_DATA_DIR, TRAIN_FILES


OUTPUT_SCALER = (
    RAW_DATA_DIR.parent
    / "processed"
    / "feature_scaler.pkl"
)

CHUNK_SIZE = 100_000


def main():
    print("=" * 70)
    print("FITTING FEATURE SCALER")
    print("=" * 70)

    scaler = StandardScaler()

    total_rows = 0

    for file_name in TRAIN_FILES:
        file_path = RAW_DATA_DIR / file_name

        print(f"\nProcessing: {file_name}")

        parquet_file = pd.read_parquet(
            file_path,
            columns=FEATURE_COLUMNS,
        )

        for start in range(
            0,
            len(parquet_file),
            CHUNK_SIZE,
        ):
            chunk = parquet_file.iloc[
                start:start + CHUNK_SIZE
            ].copy()

            chunk = chunk.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            chunk = chunk.dropna(
                subset=FEATURE_COLUMNS
            )

            if len(chunk) == 0:
                continue

            scaler.partial_fit(
                chunk[FEATURE_COLUMNS]
            )

            total_rows += len(chunk)

        del parquet_file

    print(
        f"\nRows used for scaler: {total_rows:,}"
    )

    OUTPUT_SCALER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        OUTPUT_SCALER,
    )

    print("\nScaler saved to:")
    print(OUTPUT_SCALER)

    print("\nDONE.")


if __name__ == "__main__":
    main()