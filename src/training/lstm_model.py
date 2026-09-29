import torch
import torch.nn as nn


class AttackLSTM(nn.Module):

    def __init__(
        self,
        input_size=18,
        hidden_size=128,
        num_layers=2,
        num_classes=7,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        logits = self.classifier(
            last_hidden
        )

        return logits