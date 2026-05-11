import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_nesterov(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """Nesterov: gradient is evaluated at lookahead point x_k - beta*v_k."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    traj = [x.copy()]

    for k in range(max_iter):
        lookahead = x - beta * v
        g = cg(lookahead)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        v = beta * v + alpha * g
        x = x - v
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
