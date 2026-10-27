"""Reproduce a controlled feature-scaling comparison; never run uploaded code."""
import hashlib, json, platform, warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

config = json.loads(Path(__file__).with_name('config.json').read_text())
X, y = make_classification(n_samples=600, n_features=12, n_informative=8, random_state=config['seed'])
X = X * np.array([1, 1, 1, 1, 10, 10, 100, 100, 1000, 1000, 10000, 10000])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=config['seed'], stratify=y)
params = dict(hidden_layer_sizes=(20,), random_state=config['seed'], max_iter=200, alpha=0.001)
results = {}
for label, scaled in [('baseline', False), ('scaled', True)]:
    model = make_pipeline(StandardScaler(), MLPClassifier(**params)) if scaled else MLPClassifier(**params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(X_train, y_train)
    results[label] = {'train_accuracy': float(model.score(X_train, y_train)), 'validation_accuracy': float(model.score(X_val, y_val)), 'convergence_warning': any(issubclass(w.category, ConvergenceWarning) for w in caught)}
report = {'run_type': 'measured sample', 'dataset': 'synthetic make_classification', 'config': config, 'model_parameters': params, 'python_version': platform.python_version(), 'sklearn_version': sklearn.__version__, 'train_py_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'results': results}
Path(__file__).with_name('run.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
