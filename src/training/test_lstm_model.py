import torch

from src.training.lstm_model import AttackLSTM


def main():
    print("=" * 70)
    print("LSTM FORWARD-PASS TEST")
    print("=" * 70)

    batch_size = 8
    sequence_length = 5
    feature_count = 18
    class_count = 7

    x = torch.randn(
        batch_size,
        sequence_length,
        feature_count,
    )

    model = AttackLSTM(
        input_size=feature_count,
        num_classes=class_count,
    )

    logits = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", logits.shape)

    expected_shape = (
        batch_size,
        class_count,
    )

    assert logits.shape == expected_shape

    assert torch.isfinite(logits).all()

    print("\nForward pass: OK")
    print("Output contains only finite values: OK")
    print("Shape check: OK")
    print("\nMODEL TEST PASSED.")


if __name__ == "__main__":
    main()