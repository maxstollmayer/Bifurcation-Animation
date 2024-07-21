import numpy as np
from numpy.typing import NDArray

from lib import bifurcate, bifurcate_ode


def f1(x: NDArray, a: NDArray) -> NDArray:
    return a * x * (1 - x)


def f2(x: NDArray, a: NDArray) -> NDArray:
    return a / 4 * np.sin(np.pi * x)


def f3(x: NDArray, a: NDArray) -> NDArray:
    return a / 2 * np.amin([x, 1 - x], axis=0)


def f4(x: NDArray, a: NDArray) -> NDArray:
    return x + 2 * a * (x - 1) * (2 * x - 1) * x


def f5(x: NDArray, a: NDArray) -> NDArray:
    return 1 - x - 2 * a * (x - 1) * (2 * x - 1) * x


def f6(x: NDArray, a: NDArray) -> NDArray:
    return x * x + a


def main():
    animation = bifurcate(
        f6,
        np.linspace(-2, 2, 1080),
        np.linspace(-2, 0.25, 1920),
        max_iter=24 * 10,
        alpha=0.01,
    )
    animation.save("quadratic.mp4", dpi=100, fps=24, writer="ffmpeg", bitrate=2400)


if __name__ == "__main__":
    main()

