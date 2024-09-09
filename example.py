import numpy as np
from tqdm import tqdm

from lib import Video


examples = {
    "logistic": Video(lambda x, a: a * x * (1 - x), (0, 1), (0, 4)),
    "quadratic": Video(lambda x, a: x * x + a, (-2, 2), (-2, 0.25)),
    "hat": Video(lambda x, a: a / 2 * np.amin([x, 1 - x], axis=0), (0, 1), (0, 4)),
    "cubic_positive": Video(lambda x, a: x + 2 * a * (x - 1) * (2 * x - 1) * x, (0, 1), (0, 4)),
    "cubic_negative": Video(lambda x, a: 1 - x - 2 * a * (x - 1) * (2 * x - 1) * x, (0, 1), (0, 4)),
    "sinus": Video(lambda x, a: a / 4 * np.sin(np.pi * x), (0, 1), (0, 4)),
    "logistic_ode": Video(lambda x, a: a * x * (1 - x), (0, 1), (0, 4), continuous=True),
}


def main():
    for name, video in tqdm(examples.items()):
        video.save("./examples/" + name + ".mp4")


if __name__ == "__main__":
    main()

