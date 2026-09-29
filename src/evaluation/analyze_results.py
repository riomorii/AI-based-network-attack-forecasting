import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_results.json"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "error_analysis.txt"
)

CLASS_NAMES = {
    0: "BENIGN",
    1: "BOT",
    2: "BRUTE_FORCE",
    3: "DOS",
    4: "DDOS",
    5: "INFILTRATION",
    6: "WEB_ATTACK",
}


def load_results():
    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def analyze_file(result):
    labels = result["labels_present"]
    matrix = result["confusion_matrix"]

    class_metrics = []

    for row_index, class_id in enumerate(labels):
        true_count = sum(matrix[row_index])

        correct = matrix[row_index][row_index]

        recall = (
            correct / true_count
            if true_count > 0
            else 0.0
        )

        class_metrics.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "support": true_count,
                "correct": correct,
                "recall": recall,
            }
        )

    errors = []

    for row_index, true_class in enumerate(labels):
        for column_index, predicted_class in enumerate(labels):
            if row_index == column_index:
                continue

            count = matrix[row_index][column_index]

            if count > 0:
                errors.append(
                    {
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "count": count,
                    }
                )

    errors.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    return class_metrics, errors


def build_report(results):
    lines = []

    lines.append("=" * 70)
    lines.append("NETWORK ATTACK MODEL ERROR ANALYSIS")
    lines.append("=" * 70)

    for result in results:
        lines.append("")
        lines.append("=" * 70)
        lines.append(
            f"FILE: {result['file']}"
        )
        lines.append("=" * 70)

        lines.append(
            f"Samples: {result['samples']:,}"
        )

        lines.append(
            f"Accuracy: "
            f"{result['accuracy']:.4%}"
        )

        lines.append(
            f"Balanced Accuracy: "
            f"{result['balanced_accuracy']:.4%}"
        )

        class_metrics, errors = analyze_file(
            result
        )

        lines.append("")
        lines.append("PER-CLASS RECALL")
        lines.append("-" * 70)

        for metric in class_metrics:
            lines.append(
                f"{metric['class_name']:<18}"
                f"support={metric['support']:,} "
                f"correct={metric['correct']:,} "
                f"recall={metric['recall']:.2%}"
            )

        lines.append("")
        lines.append("LARGEST MISCLASSIFICATIONS")
        lines.append("-" * 70)

        if not errors:
            lines.append(
                "No misclassifications found."
            )
        else:
            for error in errors[:10]:
                true_name = CLASS_NAMES[
                    error["true_class"]
                ]

                predicted_name = CLASS_NAMES[
                    error["predicted_class"]
                ]

                lines.append(
                    f"{true_name:<18} -> "
                    f"{predicted_name:<18} "
                    f"{error['count']:,}"
                )

    lines.append("")
    lines.append("=" * 70)
    lines.append("ANALYSIS COMPLETE")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    results = load_results()

    report = build_report(results)

    print(report)

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