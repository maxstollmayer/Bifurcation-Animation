from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Iterable
from matplotlib.artist import Artist
import numpy as np
from numpy.typing import NDArray


def bifurcate(
    f: Callable[[NDArray, NDArray], NDArray],
    xs: NDArray[np.float64],
    params: NDArray[np.float64],
    max_iter: int = 100,
    alpha: float = 0.01,
) -> FuncAnimation:
    X, A = np.meshgrid(xs, params)

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)
    ax.set_xlim(params[0], params[-1])
    ax.set_ylim(xs[0], xs[-1])
    ax.set_xlabel("parameter")
    ax.set_ylabel("position")
    plot = ax.scatter([], [], s=1, color="black", alpha=alpha, marker=".")

    def update(frame: int) -> Iterable[Artist]:
        nonlocal X
        if frame > 0:
            X = f(X, A)
        plot.set_offsets(np.c_[A.ravel(), X.ravel()])
        return []

    return FuncAnimation(fig, update, frames=max_iter, interval=1, blit=False)


def bifurcate_ode(
    f: Callable[[NDArray, NDArray], NDArray],
    xs: NDArray[np.float64],
    params: NDArray[np.float64],
    max_iter: int = 100,
    step_size: float = 0.01,
    alpha: float = 0.01,
) -> FuncAnimation:
    h = step_size

    def func(x: NDArray, a: NDArray) -> NDArray:
        k1 = f(x, a)
        k2 = f(x + h * k1 / 2, a)
        k3 = f(x + h * k2 / 2, a)
        k4 = f(x + h * k3, a)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return bifurcate(func, xs, params, max_iter, alpha)
