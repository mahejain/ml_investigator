"""Run a trusted scaling comparison on newly generated synthetic data.

This script never loads or executes an uploaded project. Results are written to
a new file, without overwriting the bundled sample records.
"""
import argparse
import json
import warnings
from pathlib import Path

def run_probe(seed=42):
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = make_classification(n_samples=600, n_features=12, n_informative=8, random_state=seed)
    X = X * np.array([1, 1, 1, 1, 10, 10, 100, 100, 1000, 1000, 10000, 10000])
    a, b, ya, yb = train_test_split(X, y, test_size=.25, random_state=seed, stratify=y)
    params = {'hidden_layer_sizes':(20,), 'random_state':seed, 'max_iter':200, 'alpha':.001}
    results = {}
    for name, scale in [('baseline', False), ('scaled', True)]:
        model = make_pipeline(StandardScaler(), MLPClassifier(**params)) if scale else MLPClassifier(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(a, ya)
        results[name] = {'train_accuracy':float(model.score(a, ya)), 'validation_accuracy':float(model.score(b, yb)), 'convergence_warning':any(issubclass(w.category, ConvergenceWarning) for w in caught)}
    return {'description':'Trusted synthetic probe; only feature scaling differs', 'seed':seed, 'results':results}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='sample_probe_results.json')
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists(): parser.error('Output exists; choose a new --output path.')
    report=run_probe(args.seed)
    with path.open('x',encoding='utf-8') as handle: json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2))
    print('Saved to',path)

if __name__ == '__main__': main()
