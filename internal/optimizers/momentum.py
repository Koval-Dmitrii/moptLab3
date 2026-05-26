# # import numpy as np
# # from typing import Callable
# #
# # from ..utils import OptResult, CallCounter
# #
# # MAX_ITER = 100_000
# #
# #
# # def gradient_descent_momentum(
# #     f: Callable, grad: Callable,
# #     x0: np.ndarray,
# #     alpha: float = 0.01,
# #     beta: float = 0.9,
# #     eps: float = 1e-8,
# #     max_iter: int = MAX_ITER,
# # ) -> OptResult:
# #     """Momentum: x_{k+1} = x_k - alpha * v_{k+1}, v_{k+1} = beta*v_k + grad f(x_k)."""
# #     cf = CallCounter(f)
# #     cg = CallCounter(grad)
# #
# #     x = x0.copy().astype(float)
# #     v = np.zeros_like(x)
# #     traj = [x.copy()]
# #
# #     for k in range(max_iter):
# #         g = cg(x)
# #         if np.linalg.norm(g) < eps:
# #             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
# #
# #         v = beta * v + g
# #         x = x - alpha * v
# #         traj.append(x.copy())
# #         cf(x)
# #
# #     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
#
# import numpy as np
# from typing import Callable
#
# from ..utils import OptResult, CallCounter
#
# MAX_ITER = 100_000
#
#
# def gradient_descent_momentum(
#     f: Callable, grad: Callable,
#     x0: np.ndarray,
#     alpha: float = 0.01,
#     beta: float = 0.9,
#     eps: float = 1e-8,
#     max_iter: int = MAX_ITER,
# ) -> OptResult:
#     """
#     Momentum (Def 7.1 из лекции):
#         g_k   = grad f(x_k)
#         m_{k+1} = beta * m_k + g_k
#         x_{k+1} = x_k - alpha * m_{k+1}
#     m_0 = 0, beta in [0, 1).
#     """
#     cf = CallCounter(f)
#     cg = CallCounter(grad)
#
#     x = x0.copy().astype(float)
#     m = np.zeros_like(x)          # m_0 = 0
#     traj = [x.copy()]
#
#     for k in range(max_iter):
#         g = cg(x)                  # g_k = grad f(x_k)
#         if np.linalg.norm(g) < eps:
#             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
#
#         m = beta * m + g           # m_{k+1} = beta*m_k + g_k
#         x = x - alpha * m          # x_{k+1} = x_k - alpha*m_{k+1}
#         traj.append(x.copy())
#         cf(x)
#
#     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


# internal/optimizers/momentum.py

import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000
MAX_GRAD_NORM = 100.0
MAX_X_NORM = 1e6


def gradient_descent_momentum(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.001,
    beta: float = 0.8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)
    m = np.zeros_like(x)

    traj = [x.copy()]

    for k in range(max_iter):

        g = cg(x)

        # gradient clipping
        g_norm = np.linalg.norm(g)
        if g_norm > MAX_GRAD_NORM:
            g = g * (MAX_GRAD_NORM / g_norm)

        m = beta * m + g

        step = alpha * m

        if np.linalg.norm(step) < eps:
            return OptResult(
                x, cf(x), k,
                cf.count, cg.count,
                True, traj
            )

        x = x - step

        # nan / inf protection
        if not np.all(np.isfinite(x)):
            return OptResult(
                x, np.inf, k,
                cf.count, cg.count,
                False, traj
            )

        # runaway protection
        if np.linalg.norm(x) > MAX_X_NORM:
            return OptResult(
                x, np.inf, k,
                cf.count, cg.count,
                False, traj
            )

        traj.append(x.copy())
        cf(x)

    return OptResult(
        x, cf(x), max_iter,
        cf.count, cg.count,
        False, traj
    )