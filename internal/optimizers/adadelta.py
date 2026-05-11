import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_adadelta(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    rho: float = 0.95,
    delta: float = 1e-6,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """AdaDelta without explicit learning rate."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    avg_g2 = np.zeros_like(x)
    avg_dx2 = np.zeros_like(x)
    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        avg_g2 = rho * avg_g2 + (1.0 - rho) * (g ** 2)
        rms_dx = np.sqrt(avg_dx2 + delta)
        rms_g = np.sqrt(avg_g2 + delta)
        dx = -(rms_dx / rms_g) * g

        x = x + dx
        avg_dx2 = rho * avg_dx2 + (1.0 - rho) * (dx ** 2)
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
