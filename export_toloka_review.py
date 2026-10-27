"""Prepare evaluation results for human review."""

import csv
import json
import sys
from pathlib import Path


if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python export_toloka_review.py "
        "evaluation_results/FILE.json"
    )

source = Path(sys.argv[1])
rows = json.loads(source.read_text(encoding="utf-8"))

output = source.with_name(
    source.stem + "_human_review.csv"
)

columns = [
    "case_id",
    "model",
    "status",
    "question",
    "evidence",
    "answer",
    "review_guidance",
    "human_score",
    "reviewer_notes",
]

with output.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for row in rows:
        evidence = "\n\n".join(
            f"{item['name']}:\n{item['content']}"
            for item in row["files"]
        )

        writer.writerow({
            "case_id": row["id"],
            "model": row.get("model", ""),
            "status": row["status"],
            "question": row["question"],
            "evidence": evidence,
            "answer": row.get("answer", row.get("error", "")),
            "review_guidance": row["review"],
            "human_score": "",
            "reviewer_notes": "",
        })

print(f"Created: {output}")