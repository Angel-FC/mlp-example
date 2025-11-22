"""
Example of MLPClassifier (Multi-Layer Perceptron) using the built-in Iris dataset.
This script trains, evaluates, and prints model performance.
All hyperparameters of the MLPClassifier are explicitly included and commented.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================
# 1. Load a public dataset (Iris)
# ============================
data = load_iris()
X = data.data
y = data.target

# Split dataset into training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================
# 2. Define MLPClassifier with ALL PARAMETERS + EXPLANATIONS
# ============================

clf = MLPClassifier(
    # ----- NETWORK ARCHITECTURE -----
    hidden_layer_sizes=(100, 50),
    # Number of neurons in each hidden layer.
    # Here: 2 hidden layers, with 100 and 50 neurons.

    activation='relu',
    # Activation function for hidden layers:
    # 'identity', 'logistic', 'tanh', 'relu'.

    solver='adam',
    # Optimization algorithm:
    # 'lbfgs' – good for small datasets
    # 'sgd' – stochastic gradient descent
    # 'adam' – recommended for most cases

    # ----- REGULARIZATION -----
    alpha=0.0001,
    # L2 regularization term to prevent overfitting.

    # ----- LEARNING SETTINGS -----
    batch_size='auto',
    # Batch size for optimization.
    # 'auto' = min(200, n_samples)

    learning_rate='constant',
    # Learning rate schedule:
    # 'constant', 'invscaling', 'adaptive'

    learning_rate_init=0.001,
    # Initial learning rate.

    power_t=0.5,
    # Exponent for 'invscaling' learning rate schedule.

    # ----- TRAINING CONTROL -----
    max_iter=500,
    # Maximum number of training iterations (epochs).

    shuffle=True,
    # Shuffle samples at each epoch.

    random_state=42,
    # Seed for reproducibility.

    tol=1e-4,
    # Minimum improvement required to continue training.

    verbose=True,
    # Print training progress.

    warm_start=False,
    # If True, reuse the solution from previous fit() calls.

    # ----- SGD-SPECIFIC PARAMETERS -----
    momentum=0.9,
    # Momentum for gradient descent updates.

    nesterovs_momentum=True,
    # Apply Nesterov momentum (faster convergence).

    early_stopping=False,
    # If True, use validation split to stop training early.

    validation_fraction=0.1,
    # Fraction of training data used as validation if early_stopping=True.

    # ----- ADAM-SPECIFIC PARAMETERS -----
    beta_1=0.9,
    # Exponential decay rate for first moment estimate.

    beta_2=0.999,
    # Exponential decay rate for second moment estimate.

    epsilon=1e-8,
    # Small constant to avoid numerical division errors.

    # ----- OTHER SETTINGS -----
    n_iter_no_change=10,
    # Stop training if no improvement over these many epochs.

    max_fun=15000
    # Maximum number of function evaluations (for solver='lbfgs')
)

# ============================
# 3. Train the neural network
# ============================
clf.fit(X_train, y_train)

# ============================
# 4. Evaluate model performance
# ============================
print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, clf.predict(X_test)))

print("\nCLASSIFICATION REPORT:")
print(classification_r_
      
