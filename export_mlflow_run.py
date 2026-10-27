"""Export selected MLflow run metadata as an investigator ZIP for local review."""
import argparse
import json
import zipfile
from pathlib import Path

def write_run_archive(run, destination):
    """Export metrics, parameters and limited provenance; never download artifacts."""
    info = run.info
    record = {
        'run_id': info.run_id,
        'experiment_id': info.experiment_id,
        'status': info.status,
        'parameters': dict(run.data.params),
        'metrics': dict(run.data.metrics),
        'code_version': run.data.tags.get('mlflow.source.git.commit'),
    }
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('mlflow_run.json', json.dumps(record, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id', help='MLflow run ID')
    parser.add_argument('--tracking-uri', default='sqlite:///mlflow.db')
    parser.add_argument('--output', default='mlflow_run_export.zip')
    args = parser.parse_args()
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        parser.error('Install MLflow first: python -m pip install mlflow')
    run = MlflowClient(tracking_uri=args.tracking_uri).get_run(args.run_id)
    destination = Path(args.output)
    if destination.exists(): parser.error('Output already exists; choose a different --output path.')
    write_run_archive(run, destination)
    print(f'Created {destination}. Review its contents for secrets before uploading.')
    print('This exports metadata only; it does not download or run artifacts.')

if __name__ == '__main__':
    main()
