from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Illustrative code: features have not yet been scaled.
model = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42)
# Training data and metrics for this example are represented in run.log.
