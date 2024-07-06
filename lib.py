from matplotlib import pyplot as plt
from typing import Callable
import numpy as np
from numpy.typing import NDArray


def calc_diagram(
    f: Callable[[float, float], float],
    xs: NDArray[np.float64] = np.linspace(0, 1, 100),
    params: NDArray[np.float64] = np.linspace(0, 4, 1000),
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(xs)
    m = len(params)
    params = np.repeat(params, n)
    xs = np.tile(xs, m)
    result = np.zeros_like(xs)

    for i in range(m * n):
        x = xs[i]
        a = params[i]
        for _ in range(max_iter):
            x = f(x, a)
            if x < 0 or x > 1:
                break
        result[i] = x

    return params, result


def plot_diagram(params, vals, s: int = 1, alpha: float = 0.5):
    plt.scatter(params, vals, s=s, alpha=alpha)
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    plt.show()
