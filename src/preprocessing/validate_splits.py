import pandas as pd

from config import LABEL_MAP
from split_config import RAW_DATA_DIR
from build_splits import split_file


def main():
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))

    all_train = []
    all_validation = []
    all_test = []

    print("=" * 70)
    print("VALIDATING 70 / 15 / 15 SPLIT")
    print("=" * 70)

    for file_path in files:
        train, validation, test = split_file(file_path)

        all_train.append(train["attack_family"])
        all_validation.append(validation["attack_family"])
        all_test.append(test["attack_family"])

    train_labels = pd.concat(
        all_train,
        ignore_index=True,
    )

    validation_labels = pd.concat(
        all_validation,
        ignore_index=True,
    )

    test_labels = pd.concat(
        all_test,
        ignore_index=True,
    )

    splits = {
        "TRAIN": train_labels,
        "VALIDATION": validation_labels,
        "TEST": test_labels,
    }

    for name, labels in splits.items():
        print(f"\n{name}")
        print("-" * 40)

        print(
            labels.value_counts().to_string()
        )

        print(
            f"\nTotal: {len(labels):,}"
        )

        print(
            "Classes:",
            sorted(labels.dropna().unique())
        )

    print("\n" + "=" * 70)
    print("CLASS COVERAGE")
    print("=" * 70)

    train_classes = set(train_labels.dropna())
    validation_classes = set(validation_labels.dropna())
    test_classes = set(test_labels.dropna())

    print(
        "TRAIN only:",
        sorted(train_classes - validation_classes - test_classes),
    )

    print(
        "VALIDATION only:",
        sorted(validation_classes - train_classes),
    )

    print(
        "TEST only:",
        sorted(test_classes - train_classes),
    )


if __name__ == "__main__":
    main()