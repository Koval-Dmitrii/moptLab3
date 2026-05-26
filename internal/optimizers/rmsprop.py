# # import numpy as np
# # from typing import Callable
# #
# # from ..utils import OptResult, CallCounter
# #
# # MAX_ITER = 100_000
# #
# #
# # def gradient_descent_rmsprop(
# #     f: Callable, grad: Callable,
# #     x0: np.ndarray,
# #     alpha: float = 0.01,
# #     rho: float = 0.9,
# #     delta: float = 1e-8,
# #     eps: float = 1e-8,
# #     max_iter: int = MAX_ITER,
# # ) -> OptResult:
# #     """RMSProp with exponential moving average of squared gradients."""
# #     cf = CallCounter(f)
# #     cg = CallCounter(grad)
# #
# #     x = x0.copy().astype(float)
# #     avg_sq = np.zeros_like(x)
# #     traj = [x.copy()]
# #
# #     for k in range(max_iter):
# #         g = cg(x)
# #         if np.linalg.norm(g) < eps:
# #             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
# #
# #         avg_sq = rho * avg_sq + (1.0 - rho) * (g ** 2)
# #         step = alpha * g / (np.sqrt(avg_sq) + delta)
# #         x = x - step
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
# def gradient_descent_rmsprop(
#     f: Callable, grad: Callable,
#     x0: np.ndarray,
#     alpha: float = 0.01,
#     rho: float = 0.9,
#     delta: float = 1e-8,
#     eps: float = 1e-8,
#     max_iter: int = MAX_ITER,
# ) -> OptResult:
#     """
#     RMSProp (Def 7.2 из лекции):
#         g_k      = grad f(x_k)
#         G_{k+1}  = rho*G_k + (1 - rho)*g_k^2    (покоординатно)
#         x_{k+1}  = x_k - alpha * g_k / (sqrt(G_{k+1}) + delta)
#     G_0 = 0, rho in [0, 1).
#     """
#     cf  = CallCounter(f)
#     cg  = CallCounter(grad)
#
#     x   = x0.copy().astype(float)
#     G   = np.zeros_like(x)          # G_0 = 0
#     traj = [x.copy()]
#
#     for k in range(max_iter):
#         g = cg(x)
#         if np.linalg.norm(g) < eps:
#             return OptResult(x, cf(x), k, cf.count, cg.count, True, traj)
#
#         G  = rho * G + (1.0 - rho) * g ** 2             # G_{k+1}
#         x  = x - alpha * g / (np.sqrt(G) + delta)        # x_{k+1}
#         traj.append(x.copy())
#         cf(x)
#
#     return OptResult(x, cf(x), max_iter, cf.count, cg.count, False, traj)


# internal/optimizers/rmsprop.py

import numpy as np
from typing import Callable

from ..utils import OptResult, CallCounter

MAX_ITER = 100_000
MAX_GRAD_NORM = 100.0
MAX_X_NORM = 1e6


def gradient_descent_rmsprop(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    alpha: float = 0.01,
    rho: float = 0.9,
    delta: float = 1e-8,
    eps: float = 1e-8,
    max_iter: int = MAX_ITER,
) -> OptResult:

    cf = CallCounter(f)
    cg = CallCounter(grad)

    x = x0.copy().astype(float)

    G = np.zeros_like(x)

    traj = [x.copy()]

    for k in range(max_iter):

        g = cg(x)

        # gradient clipping
        g_norm = np.linalg.norm(g)
        if g_norm > MAX_GRAD_NORM:
            g = g * (MAX_GRAD_NORM / g_norm)

        G = rho * G + (1.0 - rho) * g**2

        step = alpha * g / (np.sqrt(G) + delta)

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