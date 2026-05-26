import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000


def gradient_descent_adam(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    delta: float = 1e-8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:
    """
    Adam (Def 7.4 из лекции):

        g_k      = grad f(x_k)
        m_{k+1}  = beta1*m_k + (1 - beta1)*g_k
        v_{k+1}  = beta2*v_k + (1 - beta2)*g_k^2

        m_hat    = m_{k+1} / (1 - beta1^{k+1})
        v_hat    = v_{k+1} / (1 - beta2^{k+1})

        x_{k+1}  = x_k - alpha * m_hat / (sqrt(v_hat) + delta)

    m_0 = 0, v_0 = 0.
    """

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.astype(float).copy()

    m = np.zeros_like(x)
    v = np.zeros_like(x)

    traj = [x.copy()]

    for k in range(max_iter):
        t = k + 1

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

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g ** 2)

        beta1_t = beta1 ** t
        beta2_t = beta2 ** t

        m_hat = m / (1.0 - beta1_t)
        v_hat = v / (1.0 - beta2_t)

        x = x - alpha * m_hat / (np.sqrt(v_hat) + delta)

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