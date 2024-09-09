from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Iterable
from matplotlib.artist import Artist
import numpy as np
from numpy.typing import NDArray as Array
from dataclasses import dataclass

num = float | int


def bifurcate(
    f: Callable[[Array, Array], Array],
    xs: Array[np.float64],
    params: Array[np.float64],
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
    f: Callable[[Array, Array], Array],
    xs: Array[np.float64],
    params: Array[np.float64],
    max_iter: int = 100,
    step_size: float = 0.01,
    alpha: float = 0.01,
) -> FuncAnimation:
    h = step_size

    def func(x: Array, a: Array) -> Array:
        k1 = f(x, a)
        k2 = f(x + h * k1 / 2, a)
        k3 = f(x + h * k2 / 2, a)
        k4 = f(x + h * k3, a)
        return x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return bifurcate(func, xs, params, max_iter, alpha)


@dataclass
class Video:
    f: Callable[[Array, Array], Array]
    xs: tuple[num, num]
    params: tuple[num, num]
    size: tuple[int, int] = (1920, 1080)
    fps: int = 24
    secs: int = 10
    continuous: bool = False

    def animate(self) -> FuncAnimation:
        xs = np.linspace(self.xs[0], self.xs[1], self.size[0])
        params = np.linspace(self.params[0], self.params[1], self.size[1])
        max_iter = self.fps * self.secs
        alpha = max(1, min(1/max_iter, 0.01))
        if self.continuous:
            return bifurcate_ode(self.f, xs, params, max_iter=max_iter, alpha=alpha)
        return bifurcate(self.f, xs, params)

    def save(self, filename: str, dpi: int = 100) -> None:
        bitrate = self.fps * self.secs * 10
        animation = self.animate()
        animation.save(filename, dpi=dpi, fps=self.fps, writer="ffmpeg", bitrate=bitrate)
