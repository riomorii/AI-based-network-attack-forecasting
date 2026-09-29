import numpy as np
import torch
from pathlib import Path

from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

DOS2_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter_test_sequences.npz"
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

    return model


def main():
    model = load_model()

    data = np.load(DOS2_PATH)

    X = torch.from_numpy(
        data["X"]
    ).float()

    y = data["y"]

    benign_mask = y == 0

    X_benign = X[benign_mask]

    print("=" * 70)
    print("DOS2 BENIGN PREDICTION CONFIDENCE")
    print("=" * 70)

    print(
        f"DoS2 BENIGN samples: "
        f"{len(X_benign):,}"
    )

    all_confidences = []
    all_predictions = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(X_benign),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(X_benign),
            )

            logits = model(
                X_benign[start:end]
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

            all_confidences.append(
                confidence.numpy()
            )

            all_predictions.append(
                prediction.numpy()
            )

    confidences = np.concatenate(
        all_confidences
    )

    predictions = np.concatenate(
        all_predictions
    )

    unique, counts = np.unique(
        predictions,
        return_counts=True,
    )

    print()
    print("Predictions:")
    for class_id, count in zip(
        unique,
        counts,
    ):
        print(
            f"{CLASS_NAMES[class_id]:<18} "
            f"{count:,}"
        )

    print()
    print("Confidence statistics:")
    print(
        f"Mean:   {np.mean(confidences):.4f}"
    )
    print(
        f"Median: {np.median(confidences):.4f}"
    )
    print(
        f"Min:    {np.min(confidences):.4f}"
    )
    print(
        f"Max:    {np.max(confidences):.4f}"
    )

    print()
    print("Confidence percentiles:")

    for percentile in [10, 25, 50, 75, 90, 95, 99]:
        value = np.percentile(
            confidences,
            percentile,
        )

        print(
            f"P{percentile:02d}: "
            f"{value:.4f}"
        )

    print()
    print("=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()