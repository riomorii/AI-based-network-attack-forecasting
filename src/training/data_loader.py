import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_READY_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
)


class AttackDataset(Dataset):

    def __init__(self, file_path):
        data = np.load(file_path)

        self.X = torch.from_numpy(
            data["X"]
        ).float()

        self.y = torch.from_numpy(
            data["y"]
        ).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index],
        )


def create_dataloaders(
    batch_size=256,
):
    train_dataset = AttackDataset(
        MODEL_READY_DIR / "train.npz"
    )

    validation_dataset = AttackDataset(
        MODEL_READY_DIR / "validation.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        validation_loader,
    )