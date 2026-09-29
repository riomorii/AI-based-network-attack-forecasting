import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "train.npz"
)

DOS2_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter_test_sequences.npz"
)

WEB2_PATH = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter_test_sequences.npz"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "distribution_shift.txt"
)


FEATURE_NAMES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Packet Length Mean",
    "Packet Length Std",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Avg Packet Size",
    "Down/Up Ratio",
]


def load_features(path):
    data = np.load(path)
    X = data["X"].astype(np.float64)

    return X.reshape(
        -1,
        X.shape[-1],
    )


def calculate_stats(X):
    return {
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0),
        "min": np.min(X, axis=0),
        "max": np.max(X, axis=0),
    }


def build_report():
    train = load_features(TRAIN_PATH)
    dos2 = load_features(DOS2_PATH)
    web2 = load_features(WEB2_PATH)

    train_stats = calculate_stats(train)
    dos2_stats = calculate_stats(dos2)
    web2_stats = calculate_stats(web2)

    lines = []

    lines.append("=" * 100)
    lines.append("FEATURE DISTRIBUTION SHIFT ANALYSIS")
    lines.append("=" * 100)

    lines.append("")
    lines.append(
        f"Training samples: {len(train):,}"
    )

    lines.append(
        f"DoS2 samples:     {len(dos2):,}"
    )

    lines.append(
        f"Web2 samples:     {len(web2):,}"
    )

    lines.append("")
    lines.append(
        "Mean difference from training distribution"
    )
    lines.append("-" * 100)

    mean_differences = []

    for i, name in enumerate(FEATURE_NAMES):
        train_mean = train_stats["mean"][i]
        dos2_mean = dos2_stats["mean"][i]
        web2_mean = web2_stats["mean"][i]

        dos2_difference = abs(
            dos2_mean - train_mean
        )

        web2_difference = abs(
            web2_mean - train_mean
        )

        mean_differences.append(
            {
                "index": i,
                "name": name,
                "train_mean": train_mean,
                "dos2_mean": dos2_mean,
                "web2_mean": web2_mean,
                "dos2_difference": dos2_difference,
                "web2_difference": web2_difference,
            }
        )

    mean_differences.sort(
        key=lambda item: item["dos2_difference"],
        reverse=True,
    )

    for rank, item in enumerate(
        mean_differences,
        start=1,
    ):
        lines.append(
            f"{rank:02d}. "
            f"{item['name']:<30} "
            f"train={item['train_mean']:>12.5f} "
            f"dos2={item['dos2_mean']:>12.5f} "
            f"web2={item['web2_mean']:>12.5f} "
            f"dos2_diff={item['dos2_difference']:>12.5f}"
        )

    lines.append("")
    lines.append(
        "Standard deviation comparison"
    )
    lines.append("-" * 100)

    std_differences = []

    for i, name in enumerate(FEATURE_NAMES):
        train_std = train_stats["std"][i]
        dos2_std = dos2_stats["std"][i]
        web2_std = web2_stats["std"][i]

        dos2_difference = abs(
            dos2_std - train_std
        )

        web2_difference = abs(
            web2_std - train_std
        )

        std_differences.append(
            {
                "index": i,
                "name": name,
                "train_std": train_std,
                "dos2_std": dos2_std,
                "web2_std": web2_std,
                "dos2_difference": dos2_difference,
                "web2_difference": web2_difference,
            }
        )

    std_differences.sort(
        key=lambda item: item["dos2_difference"],
        reverse=True,
    )

    for rank, item in enumerate(
        std_differences,
        start=1,
    ):
        lines.append(
            f"{rank:02d}. "
            f"{item['name']:<30} "
            f"train={item['train_std']:>12.5f} "
            f"dos2={item['dos2_std']:>12.5f} "
            f"web2={item['web2_std']:>12.5f} "
            f"dos2_diff={item['dos2_difference']:>12.5f}"
        )

    lines.append("")
    lines.append("=" * 100)
    lines.append("ANALYSIS COMPLETE")
    lines.append("=" * 100)

    return "\n".join(lines)


def main():
    report = build_report()

    print(report)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print()
    print(
        f"Report saved: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()