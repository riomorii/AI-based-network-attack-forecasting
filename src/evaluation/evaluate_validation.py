import numpy as np
import torch
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "best_model.pth"
VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "validation.npz"
)

CLASS_NAMES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DOS",
    "DDOS",
    "INFILTRATION",
    "WEB_ATTACK",
]


def main():
    print("=" * 70)
    print("VALIDATION SET EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model = AttackLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    data = np.load(VALIDATION_FILE)

    X = torch.from_numpy(
        data["X"]
    ).float().to(device)

    y_true = data["y"]

    print("\nValidation samples:", len(y_true))
    print("Input shape:", X.shape)

    predictions = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            batch_size,
        ):
            batch = X[start:start + batch_size]

            logits = model(batch)

            predicted = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            predictions.append(predicted)

    y_pred = np.concatenate(predictions)

    accuracy = (
        (y_pred == y_true).mean()
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(7),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(7),
    )

    print("\nRows = True")
    print("Columns = Predicted\n")

    print(matrix)

    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()