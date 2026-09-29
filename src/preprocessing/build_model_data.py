from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.config import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    LABEL_MAP,
    LABEL_TO_ID,
)
from src.preprocessing.split_config import (
    get_train_paths,
    get_validation_paths,
    get_test_paths,
)
from src.sequences.sequence_config import (
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"

SCALER_PATH = PROCESSED_DATA_DIR / "feature_scaler.pkl"


def prepare_directories():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)


def clean_numeric_features(df):
    features = df[FEATURE_COLUMNS].copy()

    features = features.replace([np.inf, -np.inf], np.nan)

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    features = features.fillna(0.0)

    return features


def load_file(path):
    df = pd.read_parquet(path)

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {path.name}: {missing_columns}"
        )

    features = clean_numeric_features(df)

    labels = df[LABEL_COLUMN].map(LABEL_MAP)

    if labels.isna().any():
        unknown_labels = sorted(
            df.loc[labels.isna(), LABEL_COLUMN]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown labels in {path.name}: {unknown_labels}"
        )

    numeric_labels = labels.map(LABEL_TO_ID)

    if numeric_labels.isna().any():
        raise ValueError(
            f"Failed to convert labels in {path.name}"
        )

    return (
        features.to_numpy(dtype=np.float32),
        numeric_labels.to_numpy(dtype=np.int64),
    )


def fit_training_scaler(train_paths):
    scaler = StandardScaler()

    total_rows = 0

    print()
    print("=" * 70)
    print("FITTING SCALER ON TRAINING FILES ONLY")
    print("=" * 70)

    for path in train_paths:
        print(f"Fitting: {path.name}")

        features, _ = load_file(path)

        scaler.partial_fit(features)

        total_rows += len(features)

        print(f"Rows: {len(features):,}")

    print()
    print(f"Total training rows used for scaler: {total_rows:,}")
    print(f"Number of features: {scaler.n_features_in_}")

    if scaler.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError(
            "Scaler feature count does not match FEATURE_COLUMNS"
        )

    joblib.dump(scaler, SCALER_PATH)

    print(f"Scaler saved: {SCALER_PATH}")

    return scaler


def build_sequences(features, labels):
    sequence_length = SEQUENCE_LENGTH
    horizon = PREDICTION_HORIZON

    max_start = len(features) - sequence_length - horizon + 1

    if max_start <= 0:
        return (
            np.empty(
                (0, sequence_length, len(FEATURE_COLUMNS)),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    X = np.empty(
        (
            max_start,
            sequence_length,
            len(FEATURE_COLUMNS),
        ),
        dtype=np.float32,
    )

    y = np.empty(
        (max_start,),
        dtype=np.int64,
    )

    for start in range(max_start):
        end = start + sequence_length
        target_index = end + horizon - 1

        X[start] = features[start:end]
        y[start] = labels[target_index]

    return X, y


def process_split(split_name, paths, scaler):
    all_X = []
    all_y = []

    print()
    print("=" * 70)
    print(f"PROCESSING {split_name.upper()} FILES")
    print("=" * 70)

    for path in paths:
        print()
        print(f"Processing: {path.name}")

        features, labels = load_file(path)

        scaled_features = scaler.transform(features)

        scaled_features = scaled_features.astype(
            np.float32,
            copy=False,
        )

        X, y = build_sequences(
            scaled_features,
            labels,
        )

        print(f"Raw rows: {len(features):,}")
        print(f"Sequences: {len(X):,}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        output_path = (
            SEQUENCES_DIR
            / f"{path.stem}_{split_name.lower()}_sequences.npz"
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
        )

        print(f"Saved: {output_path.name}")

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError(
            f"No data produced for split: {split_name}"
        )

    combined_X = np.concatenate(
        all_X,
        axis=0,
    )

    combined_y = np.concatenate(
        all_y,
        axis=0,
    )

    print()
    print(
        f"{split_name} combined X shape: "
        f"{combined_X.shape}"
    )
    print(
        f"{split_name} combined y shape: "
        f"{combined_y.shape}"
    )

    return combined_X, combined_y


def save_model_ready(
    train_X,
    train_y,
    validation_X,
    validation_y,
):
    train_path = MODEL_READY_DIR / "train.npz"
    validation_path = MODEL_READY_DIR / "validation.npz"

    np.savez_compressed(
        train_path,
        X=train_X,
        y=train_y,
    )

    np.savez_compressed(
        validation_path,
        X=validation_X,
        y=validation_y,
    )

    print()
    print("=" * 70)
    print("MODEL-READY DATA SAVED")
    print("=" * 70)

    print(f"Train: {train_path}")
    print(f"Validation: {validation_path}")


def main():
    prepare_directories()

    train_paths = get_train_paths()
    validation_paths = get_validation_paths()
    test_paths = get_test_paths()

    print("=" * 70)
    print("CSE-CIC-IDS2018 MODEL DATA BUILDER")
    print("=" * 70)

    print()
    print(f"Training files:   {len(train_paths)}")
    print(f"Validation files: {len(validation_paths)}")
    print(f"Test files:       {len(test_paths)}")

    # -----------------------------------------------------
    # STEP 1
    # Fit scaler ONLY on training files.
    # -----------------------------------------------------
    scaler = fit_training_scaler(train_paths)

    # -----------------------------------------------------
    # STEP 2
    # Build training sequences.
    # -----------------------------------------------------
    train_X, train_y = process_split(
        "TRAIN",
        train_paths,
        scaler,
    )

    # -----------------------------------------------------
    # STEP 3
    # Build validation sequences.
    # -----------------------------------------------------
    validation_X, validation_y = process_split(
        "VALIDATION",
        validation_paths,
        scaler,
    )

    # -----------------------------------------------------
    # STEP 4
    # Build final-test sequences separately.
    #
    # These are NOT placed in model_ready/train.npz
    # or validation.npz.
    # -----------------------------------------------------
    test_X, test_y = process_split(
        "TEST",
        test_paths,
        scaler,
    )

    # -----------------------------------------------------
    # STEP 5
    # Save only train/validation for model training.
    # -----------------------------------------------------
    save_model_ready(
        train_X,
        train_y,
        validation_X,
        validation_y,
    )

    # -----------------------------------------------------
    # Final sanity checks.
    # -----------------------------------------------------
    expected_features = len(FEATURE_COLUMNS)

    if train_X.ndim != 3:
        raise ValueError(
            f"Train X must be 3D, got {train_X.ndim}D"
        )

    if validation_X.ndim != 3:
        raise ValueError(
            f"Validation X must be 3D, got {validation_X.ndim}D"
        )

    if test_X.ndim != 3:
        raise ValueError(
            f"Test X must be 3D, got {test_X.ndim}D"
        )

    if train_X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            "Train sequence length mismatch"
        )

    if validation_X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            "Validation sequence length mismatch"
        )

    if test_X.shape[1] != SEQUENCE_LENGTH:
        raise ValueError(
            "Test sequence length mismatch"
        )

    if train_X.shape[2] != expected_features:
        raise ValueError(
            "Train feature count mismatch"
        )

    if validation_X.shape[2] != expected_features:
        raise ValueError(
            "Validation feature count mismatch"
        )

    if test_X.shape[2] != expected_features:
        raise ValueError(
            "Test feature count mismatch"
        )

    if not np.isfinite(train_X).all():
        raise ValueError(
            "Train data contains non-finite values"
        )

    if not np.isfinite(validation_X).all():
        raise ValueError(
            "Validation data contains non-finite values"
        )

    if not np.isfinite(test_X).all():
        raise ValueError(
            "Test data contains non-finite values"
        )

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print("Train:")
    print(f"  X: {train_X.shape}")
    print(f"  y: {train_y.shape}")

    print()
    print("Validation:")
    print(f"  X: {validation_X.shape}")
    print(f"  y: {validation_y.shape}")

    print()
    print("Final test:")
    print(f"  X: {test_X.shape}")
    print(f"  y: {test_y.shape}")

    print()
    print("Scaler:")
    print(f"  Features: {scaler.n_features_in_}")

    print()
    print("No model training was performed.")


if __name__ == "__main__":
    main()