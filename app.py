from lib import bifurcate
import numpy as np
from numpy.typing import NDArray


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


def main():
    animation = bifurcate(
        f5, np.linspace(0, 1, 1000), np.linspace(0, 4, 4000), max_iter=24 * 10
    )
    animation.save("cubic_negative.mp4", dpi=100, fps=24, writer="ffmpeg", bitrate=2400)


if __name__ == "__main__":
    main()
