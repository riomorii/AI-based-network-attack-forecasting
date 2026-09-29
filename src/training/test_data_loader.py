import torch

from src.training.data_loader import create_dataloaders


def main():
    print("=" * 70)
    print("DATALOADER TEST")
    print("=" * 70)

    train_loader, validation_loader = create_dataloaders(
        batch_size=256
    )

    train_X, train_y = next(iter(train_loader))
    val_X, val_y = next(iter(validation_loader))

    print("\nTRAIN BATCH")
    print("X shape:", train_X.shape)
    print("y shape:", train_y.shape)
    print("X dtype:", train_X.dtype)
    print("y dtype:", train_y.dtype)

    print("\nVALIDATION BATCH")
    print("X shape:", val_X.shape)
    print("y shape:", val_y.shape)

    assert train_X.shape == (256, 5, 18)
    assert train_y.shape == (256,)

    assert val_X.shape == (256, 5, 18)
    assert val_y.shape == (256,)

    assert train_X.dtype == torch.float32
    assert train_y.dtype == torch.int64

    assert torch.isfinite(train_X).all()
    assert torch.isfinite(val_X).all()

    assert train_y.min() >= 0
    assert train_y.max() <= 6

    print("\nAll checks passed.")
    print("DATALOADER TEST PASSED.")


if __name__ == "__main__":
    main()