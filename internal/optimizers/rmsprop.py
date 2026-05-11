import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_rmsprop(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    rho: float = 0.9,
    delta: float = 1e-8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """RMSProp with exponential moving average of squared gradients."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    avg_sq = np.zeros_like(x)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        avg_sq = rho * avg_sq + (1.0 - rho) * (g ** 2)
        step = alpha * g / (np.sqrt(avg_sq) + delta)
        x = x - step
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
