# functions/himmelblau.py

import numpy as np


class Himmelblau:
    """
    Функция Химмельблау:

        f(x, y) =
            (x^2 + y - 11)^2
            + (x + y^2 - 7)^2

    Имеет 4 локальных минимума.
    """

    name: str = "Himmelblau"

    x_opt: np.ndarray = np.array([3.0, 2.0])

    def f(self, x: np.ndarray) -> float:
        return (
            (x[0] ** 2 + x[1] - 11.0) ** 2
            + (x[0] + x[1] ** 2 - 7.0) ** 2
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        dfdx = (
            4.0 * x[0] * (x[0] ** 2 + x[1] - 11.0)
            + 2.0 * (x[0] + x[1] ** 2 - 7.0)
        )

        dfdy = (
            2.0 * (x[0] ** 2 + x[1] - 11.0)
            + 4.0 * x[1] * (x[0] + x[1] ** 2 - 7.0)
        )

        return np.array([dfdx, dfdy])