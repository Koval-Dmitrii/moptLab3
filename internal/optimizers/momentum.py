import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_momentum(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    beta: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """Momentum: x_{k+1} = x_k - alpha * v_{k+1}, v_{k+1} = beta*v_k + grad f(x_k)."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        v = beta * v + g
        x = x - alpha * v
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
