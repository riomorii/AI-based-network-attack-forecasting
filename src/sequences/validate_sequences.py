import numpy as np

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"


def main():
    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    print("=" * 70)
    print("SEQUENCE VALIDATION")
    print("=" * 70)

    print(f"Files found: {len(files)}")

    for file_path in files:
        data = np.load(file_path)

        X = data["X"]
        y = data["y"]

        print(f"\n{file_path.name}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  X dtype: {X.dtype}")
        print(f"  y dtype: {y.dtype}")

        print(
            f"  X finite: "
            f"{np.isfinite(X).all()}"
        )

        print(
            f"  X min: {X.min():.4f}"
        )

        print(
            f"  X max: {X.max():.4f}"
        )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()