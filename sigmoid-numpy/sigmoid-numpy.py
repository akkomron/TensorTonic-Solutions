import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Return the sigmoid of x elementwise.

    Supports scalars, Python lists, and nested Python lists.
    """
    arr = np.asarray(x, dtype=float)
    result = 1 / (1 + np.exp(-arr))
    return float(result) if arr.ndim==0 else result
    
