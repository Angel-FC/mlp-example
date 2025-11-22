"""
Example of MLPClassifier (Multi-Layer Perceptron) using the built-in Iris dataset.
This script trains, evaluates, and prints model performance.
All hyperparameters of the MLPClassifier are explicitly included.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# 1. Load public dataset
# ============================
data = load_iris()
X = data.data
y = data.target

# Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================
# 2. Define MLPClassifier with ALL PARAMETERS
# ============================
clf = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # two hidden layers: 100 neurons + 50 neurons
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='constant',
    learning_rate_init=0.001,
    power_t=0.5,
    max_iter=500,
    shuffle=True,
    random_state=42,
    tol=1e-4,
    verbose=True,
    warm_start=False,
    momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,
    validation_fraction=0.1,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    n_iter_no_change=10,
    max_fun=15000
)

# ============================
# 3. Train model
# ============================
clf.fit(X_train, y_train)

# ============================
# 4. Evaluate model
# ============================
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, clf.predict(X_test)))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, clf.predict(X_test)))
