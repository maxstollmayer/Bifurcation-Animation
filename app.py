from lib import calc_diagram, plot_diagram
import numpy as np


def f1(x: float, a: float) -> float:
    return a * x * (1 - x)


def f2(x: float, a: float) -> float:
    return a / 4 * np.sin(np.pi * x)


def f3(x: float, a: float) -> float:
    return a / 2 * min(x, 1 - x)


def f4(x: float, a: float) -> float:
    return x + 2 * a * (x - 1) * (2 * x - 1) * x


def f5(x: float, a: float) -> float:
    return 1 - x - 2 * a * (x - 1) * (2 * x - 1) * x


def main():
    params, vals = calc_diagram(f5)
    plot_diagram(params, vals, alpha=0.1)


if __name__ == "__main__":
    main()
