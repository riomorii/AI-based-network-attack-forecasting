import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.training.data_loader import create_dataloaders
from src.training.lstm_model import AttackLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

SEED = 42

BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10
PATIENCE = 3

NUM_CLASSES = 7

# Cap the extreme inverse-frequency weights.
MAX_CLASS_WEIGHT = 3.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def calculate_class_weights():
    train_path = (
        PROJECT_ROOT
        / "data"
        / "model_ready"
        / "train.npz"
    )

    data = np.load(train_path)
    y = data["y"]

    class_counts = np.bincount(
        y,
        minlength=NUM_CLASSES,
    )

    total = len(y)

    weights = np.zeros(
        NUM_CLASSES,
        dtype=np.float32,
    )

    for class_id in range(NUM_CLASSES):
        count = class_counts[class_id]

        if count > 0:
            weights[class_id] = (
                total
                / (NUM_CLASSES * count)
            )
        else:
            weights[class_id] = 0.0

    weights = np.minimum(
        weights,
        MAX_CLASS_WEIGHT,
    )

    return (
        torch.tensor(weights, dtype=torch.float32),
        class_counts,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_loss += (
                loss.item() * X.size(0)
            )

            total_correct += (
                (predictions == y)
                .sum()
                .item()
            )

            total_samples += X.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y,
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_loss += (
            loss.item() * X.size(0)
        )

        total_correct += (
            (predictions == y)
            .sum()
            .item()
        )

        total_samples += X.size(0)

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        total_correct / total_samples
    )

    return average_loss, accuracy


def save_checkpoint(
    model,
    epoch,
    validation_loss,
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "input_size": 18,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": NUM_CLASSES,
            "dropout": 0.2,
        },
        BEST_MODEL_PATH,
    )


def main():
    set_seed(SEED)

    device = get_device()

    print("=" * 70)
    print("NETWORK ATTACK LSTM TRAINING")
    print("=" * 70)

    print()
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print()

    train_loader, validation_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(validation_loader)}"
    )

    class_weights, class_counts = (
        calculate_class_weights()
    )

    print()
    print("Class counts:")

    for class_id, count in enumerate(
        class_counts
    ):
        print(
            f"  Class {class_id}: {count:,}"
        )

    print()
    print("Capped class weights:")

    for class_id, weight in enumerate(
        class_weights
    ):
        print(
            f"  Class {class_id}: "
            f"{weight.item():.4f}"
        )

    class_weights = class_weights.to(
        device
    )

    model = AttackLSTM(
        input_size=18,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
        )

        validation_loss, validation_accuracy = (
            evaluate(
                model,
                validation_loader,
                criterion,
                device,
            )
        )

        improved = (
            validation_loss
            < best_validation_loss
        )

        if improved:
            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                epoch,
                validation_loss,
            )

            status = "Best"

        else:
            epochs_without_improvement += 1

            status = (
                f"No improvement "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2%} | "
            f"{status}"
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print()
            print(
                "Early stopping triggered."
            )
            break

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Best model: {BEST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()