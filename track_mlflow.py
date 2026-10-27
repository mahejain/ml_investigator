"""Import the existing sample results into MLflow."""

import json
from pathlib import Path

import mlflow


# Read the results of our existing sample experiment.
report_path = Path(__file__).parent / "sample_project" / "run.json"
report = json.loads(report_path.read_text(encoding="utf-8"))

# Store the MLflow tracking database locally.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("ML investigator sample")


# Record baseline and scaled results as separate MLflow runs.
for label, metrics in report["results"].items():
    with mlflow.start_run(run_name=label):

        mlflow.log_params({
            "scaling": label,
            "seed": report["config"]["seed"],
            "max_iter": report["model_parameters"]["max_iter"],
            "alpha": report["model_parameters"]["alpha"],
            "hidden_layer_sizes": str(
                report["model_parameters"]["hidden_layer_sizes"]
            ),
            "sklearn_version": report["sklearn_version"],
            "python_version": report["python_version"],
            "train_py_sha256": report["train_py_sha256"],
            "dataset": report["dataset"],
        })

        mlflow.log_metrics({
            "train_accuracy": metrics["train_accuracy"],
            "validation_accuracy": metrics["validation_accuracy"],
            "convergence_warning": int(metrics["convergence_warning"]),
        })

        mlflow.log_artifact(str(report_path))


print("Imported both existing sample results into MLflow.")