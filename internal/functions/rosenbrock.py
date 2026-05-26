import numpy as np


class Rosenbrock:
    """
    Функция Розенброка:

        f(x, y) =
            (1 - x)^2 + 100(y - x^2)^2

    Глобальный минимум:
        f(1, 1) = 0
    """

    name: str = "Rosenbrock"

    x_opt: np.ndarray = np.array([1.0, 1.0])

    def f(self, x: np.ndarray) -> float:
        return (
            (1.0 - x[0]) ** 2
            + 100.0 * (x[1] - x[0] ** 2) ** 2
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = (
            -2.0 * (1.0 - x[0])
            - 400.0 * x[0] * (x[1] - x[0] ** 2)
        )

        dfdy = (
            200.0 * (x[1] - x[0] ** 2)
        )

        return np.array([dfdx, dfdy])