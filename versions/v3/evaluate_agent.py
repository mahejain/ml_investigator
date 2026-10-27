"""Run three synthetic review cases against the investigator."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import app


CASES = [
    {
        "id": "generalization_gap",
        "question": "What might explain this gap, and what should I test next?",
        "files": [
            {
                "name": "run.log",
                "content": (
                    "train_accuracy: 0.96\n"
                    "validation_accuracy: 0.70\n"
                ),
            },
            {
                "name": "model.py",
                "content": (
                    "from sklearn.neural_network import MLPClassifier\n"
                    "model = MLPClassifier(hidden_layer_sizes=(100, 100))\n"
                ),
            },
        ],
        "review": (
            "Does it cite the two accuracies and describe overfitting "
            "as a hypothesis, not a proven cause? "
            "Does it propose a controlled test?"
        ),
    },
    {
        "id": "no_metrics",
        "question": "Why did this model fail?",
        "files": [
            {
                "name": "model.py",
                "content": (
                    "from sklearn.linear_model import LogisticRegression\n"
                    "model = LogisticRegression()\n"
                ),
            },
            {
                "name": "notes.txt",
                "content": (
                    "No training or validation metrics were recorded.\n"
                ),
            },
        ],
        "review": (
            "Does it say the outcome and cause are unknown, "
            "and ask for missing evidence instead of inventing results?"
        ),
    },
    {
        "id": "scaling_hypothesis",
        "question": (
            "What might improve the model, and how could I test it?"
        ),
        "files": [
            {
                "name": "train.py",
                "content": (
                    "from sklearn.neural_network import MLPClassifier\n"
                    "model = MLPClassifier(random_state=42)\n"
                    "# No scaling is shown in this excerpt.\n"
                ),
            },
            {
                "name": "run.log",
                "content": (
                    "train_accuracy: 0.55\n"
                    "validation_accuracy: 0.53\n"
                ),
            },
        ],
        "review": (
            "Does it distinguish missing scaling in the excerpt from "
            "proof about the full run? Does it suggest a fair test "
            "without claiming scaling must help?"
        ),
    },
]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_folder = Path("evaluation_results")
    output_folder.mkdir(exist_ok=True)

    results = []

    for case in CASES:
        print("Running:", case["id"], flush=True)

        row = {
            "id": case["id"],
            "question": case["question"],
            "review": case["review"],
            "files": case["files"],
            "status": "ok",
        }

        try:
            response = app.investigate(case["files"], case["question"])

            row.update({
                "answer": response["answer"],
                "citations": response["citation_audit"],
                "incomplete": response.get("incomplete", False),
                "warning": response.get("warning"),
            })

        except Exception as exc:
            row.update({
                "status": "error",
                "error": str(exc),
            })

        results.append(row)

    json_path = output_folder / f"evaluation_{stamp}.json"
    csv_path = output_folder / f"review_{stamp}.csv"

    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    columns = [
        "id",
        "status",
        "question",
        "review",
        "answer",
        "human_score",
        "reviewer_notes",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for row in results:
            writer.writerow({
                column: row.get(column, "")
                for column in columns
            })

    print(f"\nFull results: {json_path}")
    print(f"Review sheet: {csv_path}")
    print(
        "These are synthetic test cases. "
        "Review the answers before drawing conclusions."
    )


if __name__ == "__main__":
    main()