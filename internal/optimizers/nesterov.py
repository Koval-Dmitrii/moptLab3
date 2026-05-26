import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000
MAX_GRAD_NORM = 100.0
MAX_X_NORM = 1e6


def gradient_descent_nesterov(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    x_prev = x.copy()

    traj = [x.copy()]

    for k in range(max_iter):

        y = x + beta * (x - x_prev)

        g = cg(y)

        # gradient clipping
        g_norm = np.linalg.norm(g)
        if g_norm > MAX_GRAD_NORM:
            g = g * (MAX_GRAD_NORM / g_norm)

        step = alpha * g

        if np.linalg.norm(step) < eps:
            return OptResult(
                x, cf(x), k,
                cf.count, cg.count,
                True, traj
            )

        x_next = y - step

        # nan / inf protection
        if not np.all(np.isfinite(x_next)):
            return OptResult(
                x_next, np.inf, k,
                cf.count, cg.count,
                False, traj
            )

        # runaway protection
        if np.linalg.norm(x_next) > MAX_X_NORM:
            return OptResult(
                x_next, np.inf, k,
                cf.count, cg.count,
                False, traj
            )

        x_prev = x.copy()
        x = x_next

        traj.append(x.copy())
        cf(x)

    return OptResult(
        x, cf(x), max_iter,
        cf.count, cg.count,
        False, traj
    )