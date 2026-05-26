# # import numpy as np
# # from typing import Callable
# #
# # from ..utils import OptResult, CallCounter
# #
# # MAX_ITER = 100_000
# #
# #
# # def gradient_descent_nesterov(
# #     f: Callable, grad: Callable,
# #     x0: np.ndarray,
# #     alpha: float = 0.01,
# #     beta: float = 0.9,
# #     eps: float = 1e-8,
# #     max_iter: int = MAX_ITER,
# # ) -> OptResult:
# #     """Nesterov: gradient is evaluated at lookahead point x_k - beta*v_k."""
# #     cf = CallCounter(f)
# #     cg = CallCounter(grad)
# #
# #     x = x0.copy().astype(float)
# #     v = np.zeros_like(x)
# #     traj = [x.copy()]
# #
# #     for k in range(max_iter):
# #         lookahead = x - beta * v
# #         g = cg(lookahead)
# #         if np.linalg.norm(g) < eps:
# #             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
# #
# #         v = beta * v + alpha * g
# #         x = x - v
# #         traj.append(x.copy())
# #         cf(x)
# #
# #     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)
# import numpy as np
# from typing import Callable
#
# from ..utils import OptResult, CallCounter
#
# MAX_ITER = 100_000
#
#
# def gradient_descent_nesterov(
#     f: Callable, grad: Callable,
#     x0: np.ndarray,
#     alpha: float = 0.01,
#     beta: float = 0.9,
#     eps: float = 1e-8,
#     max_iter: int = MAX_ITER,
# ) -> OptResult:
#     """
#     Nesterov momentum (Def 7.2 из лекции):
#         y_k   = x_k + beta * (x_k - x_{k-1})
#         x_{k+1} = y_k - alpha * grad f(y_k)
#     beta in [0, 1). Инициализация: x_{-1} = x_0 (нет предыдущей точки).
#     """
#     cf = CallCounter(f)
#     cg = CallCounter(grad)
#
#     x      = x0.copy().astype(float)
#     x_prev = x.copy()              # x_{-1} = x_0
#     traj   = [x.copy()]
#
#     for k in range(max_iter):
#         y = x + beta * (x - x_prev)   # y_k = x_k + beta*(x_k - x_{k-1})
#         g = cg(y)                      # grad f(y_k)
#
#         if np.linalg.norm(g) < eps:
#             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
#
#         x_next = y - alpha * g         # x_{k+1} = y_k - alpha*grad f(y_k)
#         x_prev = x.copy()
#         x      = x_next
#
#         traj.append(x.copy())
#         cf(x)
#
#     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


# internal/optimizers/nesterov.py

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