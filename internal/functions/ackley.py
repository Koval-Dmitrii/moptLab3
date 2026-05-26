import numpy as np


class Ackley:
    """
    Функция Экли:

        f(x, y) =
            -20 * exp(-0.2 * sqrt(0.5 * (x² + y²)))
            - exp(0.5 * (cos(2πx) + cos(2πy)))
            + 20 + e

    Глобальный минимум:
        f(0, 0) = 0
    """

    name: str = "Ackley"

    x_opt: np.ndarray = np.array([0.0, 0.0])

    def f(self, x: np.ndarray) -> float:
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi

        radius = np.sqrt(0.5 * np.sum(x ** 2))

        cosine_term = 0.5 * np.sum(np.cos(c * x))

        return (
            -a * np.exp(-b * radius)
            - np.exp(cosine_term)
            + a
            + np.e
        )

    def grad(self, x: np.ndarray) -> np.ndarray:
        a = 20.0
        b = 0.2
        c = 2.0 * np.pi

        radius = np.sqrt(0.5 * np.sum(x ** 2))

        if radius < 1e-12:
            term1 = np.zeros_like(x)
        else:
            term1 = (
                (a * b / (2.0 * radius))
                * x
                * np.exp(-b * radius)
            )

        cosine_term = (
            0.5 * np.sum(np.cos(c * x))
        )

        term2 = (
            np.pi
            * np.sin(c * x)
            * np.exp(cosine_term)
        )

        return term1 + term2