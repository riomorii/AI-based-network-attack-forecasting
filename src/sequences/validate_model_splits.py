import numpy as np
from pathlib import Path

from src.preprocessing.config import LABEL_TO_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"


ID_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_ID.items()
}


def inspect_split(name):
    path = MODEL_READY_DIR / f"{name}.npz"

    data = np.load(path)

    X = data["X"]
    y = data["y"]

    print(f"\n{name.upper()}")
    print("-" * 50)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X dtype:", X.dtype)
    print("y dtype:", y.dtype)
    print("X finite:", np.isfinite(X).all())

    print("\nClasses:")

    for class_id in sorted(np.unique(y)):
        count = np.sum(y == class_id)
        label = ID_TO_LABEL[class_id]

        print(
            f"  {class_id}: "
            f"{label:<15} "
            f"{count:,}"
        )


def main():
    print("=" * 70)
    print("MODEL SPLIT VALIDATION")
    print("=" * 70)

    inspect_split("train")
    inspect_split("validation")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()