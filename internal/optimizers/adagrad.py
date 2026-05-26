# optimizers/adagrad.py

import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_adagrad(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.1,
    delta: float = 1e-8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """
    AdaGrad:

        g_k      = grad f(x_k)
        G_{k+1}  = G_k + g_k^2

        x_{k+1} =
            x_k - alpha * g_k / (sqrt(G_{k+1}) + delta)

    G_0 = 0.
    """

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.astype(float).copy()

    grad_sq_sum = np.zeros_like(x)

    traj = [x.copy()]

    for k in range(max_iter):
        g = cg(x)

        if np.linalg.norm(g) < eps:
            return OptResult(
                x,
                cf(x),
                k,
                cf.count,
                cg.count,
                True,
                traj,
            )

        grad_sq_sum += g ** 2

        step = (
            alpha * g
            / (np.sqrt(grad_sq_sum) + delta)
        )

        x = x - step

        traj.append(x.copy())

    return OptResult(
        x,
        cf(x),
        max_iter,
        cf.count,
        cg.count,
        False,
        traj,
    )