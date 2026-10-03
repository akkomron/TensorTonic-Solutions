import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    num_samples, num_features = X.shape # samples, features
    y = y.reshape(-1) # flatten it

    w = np.zeros(num_features) # broadcastable to X
    b = 0.0

    for _ in range(steps):

        z = np.dot(X, w) + b

        y_pred = _sigmoid(z)

        error = y_pred - y

        dw = (1 / num_samples) * np.dot(X.T, error)
        db = (1 / num_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return (w, float(b))
    
    