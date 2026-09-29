import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocessing.config import LABEL_MAP, LABEL_TO_ID
from src.preprocessing.split_config import TRAIN_FILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


def load_sequence_file(file_path):
    data = np.load(file_path)

    return data["X"], data["y"]


def convert_labels(labels):
    normalized_labels = []

    for label in labels:
        normalized_labels.append(
            LABEL_MAP[label]
        )

    converted_labels = []

    for label in normalized_labels:
        converted_labels.append(
            LABEL_TO_ID[label]
        )

    return np.array(
        converted_labels,
        dtype=np.int64,
    )

def main():
    print("=" * 70)
    print("BUILDING MODEL-READY SPLITS")
    print("=" * 70)

    train_X_parts = []
    train_y_parts = []

    for file_name in TRAIN_FILES:
        sequence_name = (
            Path(file_name).stem
            + "_sequences.npz"
        )

        file_path = SEQUENCE_DIR / sequence_name

        print(f"\nLoading: {sequence_name}")

        X, y = load_sequence_file(file_path)

        y = convert_labels(y)

        print(f"  X: {X.shape}")
        print(f"  y: {y.shape}")

        train_X_parts.append(X)
        train_y_parts.append(y)

    X = np.concatenate(
        train_X_parts,
        axis=0,
    )

    y = np.concatenate(
        train_y_parts,
        axis=0,
    )

    print("\nCombined development data:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_DIR / "train.npz",
        X=X_train,
        y=y_train,
    )

    np.savez_compressed(
        OUTPUT_DIR / "validation.npz",
        X=X_validation,
        y=y_validation,
    )

    print("\nFinal splits:")
    print(f"  Train X:      {X_train.shape}")
    print(f"  Train y:      {y_train.shape}")
    print(f"  Validation X: {X_validation.shape}")
    print(f"  Validation y: {y_validation.shape}")

    print("\nClass distribution:")

    for class_id in sorted(
        np.unique(y)
    ):
        total = np.sum(y == class_id)
        train_count = np.sum(y_train == class_id)
        val_count = np.sum(y_validation == class_id)

        print(
            f"  Class {class_id}: "
            f"total={total:,} | "
            f"train={train_count:,} | "
            f"validation={val_count:,}"
        )

    print("\nDONE.")


if __name__ == "__main__":
    main()