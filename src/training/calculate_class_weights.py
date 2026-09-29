import numpy as np
from pathlib import Path

from sklearn.utils.class_weight import compute_class_weight

from src.preprocessing.config import LABEL_TO_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "train.npz"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "class_weights.npy"
)


ID_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_ID.items()
}


def main():
    print("=" * 70)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 70)

    data = np.load(TRAIN_FILE)

    y = data["y"]

    classes = np.arange(
        len(LABEL_TO_ID)
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    weights = weights.astype(
        np.float32
    )

    print("\nClass weights:")

    for class_id, weight in zip(
        classes,
        weights,
    ):
        print(
            f"  {class_id}: "
            f"{ID_TO_LABEL[class_id]:<15} "
            f"{weight:.6f}"
        )

    np.save(
        OUTPUT_FILE,
        weights,
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print("\nDONE.")


if __name__ == "__main__":
    main()