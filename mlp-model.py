"""
--------------------------------------------------------------------------
DETAILED MLP CLASSIFIER EXAMPLE 
--------------------------------------------------------------------------

This script trains an MLPClassifier (Multi-Layer Perceptron Neural Network)
using the public Iris dataset available directly inside scikit-learn.

"""

# =======================================================================
# 0. IMPORT ALL REQUIRED LIBRARIES
# =======================================================================

# Dataset loader
from sklearn.datasets import load_iris

# Training/testing split
from sklearn.model_selection import train_test_split

# MLP neural network classifier
from sklearn.neural_network import MLPClassifier

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix


# =======================================================================
# 1. LOAD A PUBLIC DATASET (IRIS)
# =======================================================================

"""
load_iris() loads the classical Iris Flower dataset.
It contains:
- 150 samples (rows)
- 4 numerical features (columns):
    * Sepal Length
    * Sepal Width
    * Petal Length
    * Petal Width
- 3 classes of iris species:
    * 0 → Setosa
    * 1 → Versicolor
    * 2 → Virginica
"""

data = load_iris()

# Feature matrix (shape: 150 x 4)
X = data.data

# Target labels (shape: 150)
y = data.target

print("Dataset loaded successfully.")
print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)


# =======================================================================
# 2. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# =======================================================================

"""
We split the dataset so we can evaluate generalization:
- Training set: 70% (105 samples)
- Test set:     30% (45 samples)

stratify=y ensures each class appears proportionally in both sets.

random_state=42 ensures reproducibility:
    every time you run the code, you get the same split.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nData split complete.")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =======================================================================
# 3. DEFINE THE MLP CLASSIFIER (EXPLAIN EVERY PARAMETER)
# =======================================================================

"""
MLPClassifier trains a fully connected feed-forward artificial neural network.

Below we specify EVERY MAJOR PARAMETER.
"""

clf = MLPClassifier(
    # ------------------------------------------------------------------
    # (A) HIDDEN LAYER ARCHITECTURE
    # ------------------------------------------------------------------
    hidden_layer_sizes=(100, 50),
    # Defines the network architecture:
    # - The tuple means: 2 hidden layers.
    # - First hidden layer → 100 neurons.
    # - Second hidden layer → 50 neurons.
    # More layers/neurons = more representational power but risk of overfitting.

    # ------------------------------------------------------------------
    # (B) ACTIVATION FUNCTION FOR EACH NEURON
    # ------------------------------------------------------------------
    activation='relu',
    # Activation function applied to each hidden layer:
    # Options: 'identity', 'logistic', 'tanh', 'relu'.
    # RELU is common: trains fast and helps with vanishing gradient issues.

    # ------------------------------------------------------------------
    # (C) SOLVER (OPTIMIZATION ALGORITHM)
    # ------------------------------------------------------------------
    solver='adam',
    # Optimization algorithm used to update the network weights:
    # 'lbfgs' (quasi-Newton, good for small datasets),
    # 'sgd' (stochastic gradient descent),
    # 'adam' (adaptive moment estimation, robust default).

    # ------------------------------------------------------------------
    # (D) REGULARIZATION TO PREVENT OVERFITTING
    # ------------------------------------------------------------------
    alpha=0.0001,
    # L2 regularization strength (weight decay). Higher alpha => stronger regularization.

    # ------------------------------------------------------------------
    # (E) MINI-BATCH SETTINGS
    # ------------------------------------------------------------------
    batch_size='auto',
    # Batch size for optimization. 'auto' -> min(200, n_samples).

    # ------------------------------------------------------------------
    # (F) LEARNING RATE STRATEGY
    # ------------------------------------------------------------------
    learning_rate='constant',
    # Learning rate schedule: 'constant', 'invscaling', or 'adaptive'.

    learning_rate_init=0.001,
    # Initial learning rate (for 'sgd' and 'adam').

    power_t=0.5,
    # Exponent for inverse scaling (used when learning_rate='invscaling').

    # ------------------------------------------------------------------
    # (G) TRAINING PARAMETERS
    # ------------------------------------------------------------------
    max_iter=500,
    # Maximum number of iterations (epochs).

    shuffle=True,
    # Shuffle samples in each epoch.

    random_state=42,
    # Random seed for reproducible weight initialization and shuffling.

    tol=1e-4,
    # Tolerance for optimization: minimum improvement to continue.

    verbose=True,
    # If true, prints progress messages during training.

    warm_start=False,
    # If true, reuse previous solution when calling fit() again.

    # ------------------------------------------------------------------
    # (H) SGD-SPECIFIC PARAMETERS (ONLY USED IF solver='sgd')
    # ------------------------------------------------------------------
    momentum=0.9,
    # Momentum for gradient descent (only relevant for solver='sgd').

    nesterovs_momentum=True,
    # Whether to use Nesterov's momentum (only for 'sgd').

    # ------------------------------------------------------------------
    # (I) EARLY STOPPING SETTINGS
    # ------------------------------------------------------------------
    early_stopping=False,
    # If true, automatically stops training when validation score is not improving.

    validation_fraction=0.1,
    # Fraction of training set to hold out as validation for early stopping.

    # ------------------------------------------------------------------
    # (J) ADAM OPTIMIZER INTERNAL PARAMETERS
    # ------------------------------------------------------------------
    beta_1=0.9,
    # Exponential decay rate for the first moment estimates (Adam).

    beta_2=0.999,
    # Exponential decay rate for the second moment estimates (Adam).

    epsilon=1e-8,
    # Small epsilon to avoid numerical issues in Adam.

    # ------------------------------------------------------------------
    # (K) CONVERGENCE BEHAVIOR
    # ------------------------------------------------------------------
    n_iter_no_change=10,
    # Number of epochs with no improvement to wait before stopping (if applicable).

    max_fun=15000
    # Maximum number of function evaluations (used by 'lbfgs' solver).
)

print("\nMLP model defined with all parameters.")


# =======================================================================
# 4. TRAIN THE NEURAL NETWORK
# =======================================================================

"""
The .fit() method performs:
1. Forward propagation: compute activations through the network
2. Backpropagation: compute gradients of the loss wrt weights
3. Weight update using the selected optimizer (here: Adam)
4. Repeat until convergence or max_iter

Because verbose=True, training progress will be printed to the console.
"""

print("\nTraining the neural network...")
clf.fit(X_train, y_train)
print("\nTraining complete.")


# =======================================================================
# 5. MAKE PREDICTIONS AND EVALUATE PERFORMANCE
# =======================================================================

"""
We evaluate the trained network on the unseen test set.
We compute a confusion matrix and a classification report
(precision, recall, f1-score for each class).
"""

y_pred = clf.predict(X_test)

print("\n=========================== CONFUSION MATRIX ===========================")
print(confusion_matrix(y_test, y_pred))

print("\n======================== CLASSIFICATION REPORT =========================")
print(classification_report(y_test, y_pred))
