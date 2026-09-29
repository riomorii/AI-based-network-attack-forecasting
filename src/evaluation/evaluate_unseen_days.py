import numpy as np
import torch
from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from src.preprocessing.config import LABEL_MAP, LABEL_TO_ID
from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "best_model.pth"
SEQUENCE_DIR = PROJECT_ROOT / "data" / "sequences"

TEST_FILES = [
    "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter_sequences.npz",
    "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter_sequences.npz",
]

CLASS_NAMES = [
    "BENIGN",
    "BOT",
    "BRUTE_FORCE",
    "DOS",
    "DDOS",
    "INFILTRATION",
    "WEB_ATTACK",
]


def load_model(device):
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

    return model


def convert_labels(labels):
    converted_labels = []

    for label in labels:
        normalized_label = LABEL_MAP[label]
        numeric_label = LABEL_TO_ID[normalized_label]
        converted_labels.append(numeric_label)

    return np.array(
        converted_labels,
        dtype=np.int64,
    )


def evaluate_file(model, file_path, device):
    print("\n" + "=" * 70)
    print(f"TEST FILE: {file_path.name}")
    print("=" * 70)

    data = np.load(file_path)

    X = torch.from_numpy(
        data["X"]
    ).float().to(device)

    y_true = convert_labels(
        data["y"]
    )

    predictions = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(X),
            batch_size,
        ):
            batch = X[
                start:start + batch_size
            ]

            logits = model(batch)

            predicted = (
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

            predictions.append(predicted)

    y_pred = np.concatenate(
        predictions
    ).astype(np.int64)

    accuracy = (
        y_pred == y_true
    ).mean()

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    print("\nSamples:", len(y_true))

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

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

    print("CONFUSION MATRIX")
    print("-" * 70)

    print("Rows = True")
    print("Columns = Predicted\n")

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(7),
    )

    print(matrix)


def main():
    print("=" * 70)
    print("UNSEEN-DAY EVALUATION")
    print("=" * 70)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    model = load_model(device)

    for filename in TEST_FILES:

        file_path = SEQUENCE_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing sequence file:\n{file_path}"
            )

        evaluate_file(
            model,
            file_path,
            device,
        )

    print("\n" + "=" * 70)
    print("UNSEEN-DAY EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()