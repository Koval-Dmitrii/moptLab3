import numpy as np


class QuadraticFunction:
    """
    Базовая квадратичная функция:

        f(x) = 0.5 * x^T A x
    """

    name: str = "Quadratic"

    A: np.ndarray

    x_opt: np.ndarray = np.zeros(2)

    def f(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.A @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def hessian(self) -> np.ndarray:
        return self.A


class QuadraticWellConditioned(QuadraticFunction):
    """
    Хорошо обусловленная квадратичная функция.

    κ = 1
    """

    name: str = "Quadratic (well-cond, κ=1)"

    A = np.array([
        [2.0, 0.0],
        [0.0, 2.0],
    ])


class QuadraticIllConditioned(QuadraticFunction):
    """
    Плохо обусловленная квадратичная функция.

    κ = 100
    """

    name: str = "Quadratic (ill-cond, κ=100)"

    A = np.array([
        [100.0, 0.0],
        [0.0, 1.0],
    ])