# optimizers/adadelta.py

import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_adadelta(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    rho: float = 0.95,
    delta: float = 1e-6,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """
    AdaDelta (Def 7.3 из лекции):

        g_k = grad f(x_k)

        G_{k+1} =
            rho*G_k + (1-rho)*g_k^2

        Delta x_k =
            -sqrt(u_k + delta)
             / sqrt(G_{k+1} + delta) * g_k

        u_{k+1} =
            rho*u_k + (1-rho)*(Delta x_k)^2

        x_{k+1} = x_k + Delta x_k

    G_0 = 0, u_0 = 0.
    """

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.astype(float).copy()

    avg_grad_sq = np.zeros_like(x)
    avg_step_sq = np.zeros_like(x)

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

        avg_grad_sq = (
            rho * avg_grad_sq
            + (1.0 - rho) * (g ** 2)
        )

        step = -(
            np.sqrt(avg_step_sq + delta)
            / np.sqrt(avg_grad_sq + delta)
        ) * g

        avg_step_sq = (
            rho * avg_step_sq
            + (1.0 - rho) * (step ** 2)
        )

        x = x + step

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