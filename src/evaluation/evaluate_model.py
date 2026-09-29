import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
RESULTS_PATH = RESULTS_DIR / "evaluation_results.json"

CLASS_NAMES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DOS",
    "DDOS",
    "INFILTRATION",
    "WEB_ATTACK",
]


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = AttackLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, checkpoint


def evaluate_file(model, path):
    data = np.load(path)

    X = torch.from_numpy(
        data["X"]
    ).float()

    y = data["y"]

    predictions = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(X),
            )

            logits = model(
                X[start:end]
            )

            batch_predictions = (
                torch.argmax(
                    logits,
                    dim=1,
                )
                .numpy()
            )

            predictions.append(
                batch_predictions
            )

    predictions = np.concatenate(
        predictions
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y,
            predictions,
        )
    )

    labels_present = sorted(
        np.unique(
            np.concatenate(
                [y, predictions]
            )
        ).tolist()
    )

    target_names = [
        CLASS_NAMES[i]
        for i in labels_present
    ]

    report = classification_report(
        y,
        predictions,
        labels=labels_present,
        target_names=target_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=labels_present,
    )

    return (
        len(y),
        accuracy,
        balanced_accuracy,
        report,
        matrix,
        labels_present,
    )


def main():
    print("=" * 70)
    print("NETWORK ATTACK MODEL EVALUATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, checkpoint = load_model()

    print()
    print(f"Model: {MODEL_PATH}")
    print(
        f"Best training epoch: "
        f"{checkpoint['epoch']}"
    )
    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    test_files = sorted(
        SEQUENCES_DIR.glob(
            "*_test_sequences.npz"
        )
    )

    if not test_files:
        raise FileNotFoundError(
            "No test sequence files found."
        )

    print()
    print(
        f"Test files found: "
        f"{len(test_files)}"
    )

    results = []

    for path in test_files:
        print()
        print("=" * 70)
        print(
            f"TEST FILE: {path.name}"
        )
        print("=" * 70)

        (
            sample_count,
            accuracy,
            balanced_accuracy,
            report,
            matrix,
            labels_present,
        ) = evaluate_file(
            model,
            path,
        )

        print()
        print(
            f"Samples: {sample_count:,}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4%}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4%}"
        )

        print()
        print("Classification Report:")
        print(report)

        print("Confusion Matrix:")

        print(
            "Labels:",
            [
                CLASS_NAMES[i]
                for i in labels_present
            ],
        )

        print(matrix)

        results.append(
            {
                "file": path.name,
                "samples": sample_count,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "labels_present": labels_present,
                "confusion_matrix": matrix.tolist(),
                "classification_report": report,
            }
        )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print(
        f"Results saved: {RESULTS_PATH}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()