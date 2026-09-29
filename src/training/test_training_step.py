import numpy as np
import torch
import torch.nn as nn

from src.training.data_loader import create_dataloaders
from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

WEIGHTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "class_weights.npy"
)


def main():
    print("=" * 70)
    print("REAL TRAINING-STEP TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    train_loader, _ = create_dataloaders(
        batch_size=256
    )

    X, y = next(iter(train_loader))

    X = X.to(device)
    y = y.to(device)

    weights = np.load(
        WEIGHTS_FILE
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    model = AttackLSTM(
        input_size=18,
        num_classes=7,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    optimizer.zero_grad()

    logits = model(X)

    loss = loss_function(
        logits,
        y,
    )

    print("\nInput shape:", X.shape)
    print("Target shape:", y.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    assert logits.shape == (256, 7)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()

    optimizer.step()

    print("\nBackward pass: OK")
    print("Optimizer step: OK")
    print("\nTRAINING STEP TEST PASSED.")


if __name__ == "__main__":
    main()