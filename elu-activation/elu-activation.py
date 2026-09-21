import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    return [x if x > 0 else alpha * (math.exp(x) - 1) for x in x]