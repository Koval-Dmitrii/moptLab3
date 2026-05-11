import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_adam(
    f: Callable, grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    delta: float = 1e-8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """Adam with bias correction."""
    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    traj = [x.copy()]

    for k in range(max_iter):
        t = k + 1
        g = cg(x)
        if np.linalg.norm(g) < eps:
            return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)

        x = x - alpha * m_hat / (np.sqrt(v_hat) + delta)
        traj.append(x.copy())
        cf(x)

    return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
