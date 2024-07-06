import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
m = 1000
n = 101
max_iter = 100


def f(x, a):
    return a * x * (1 - x)


params = np.repeat(np.linspace(0, 4, m), n)
xs = np.tile(np.linspace(0, 1, n), m)

fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
ax.set_xlim(params[0], params[-1])
ax.set_ylim(xs[0], xs[-1])
ax.set_xlabel("a")
ax.set_ylabel("x")

artists = []


def animate(frame: int):
    global artists

    i = frame // max_iter

    xs[i] = f(xs[i], params[i])

    for artist in artists:
        artist.remove()

    settled_points = ax.scatter(params[:i], xs[:i], s=1, color="black")
    current_point = ax.scatter(params[i], xs[i], s=10, color="red")

    artists = [settled_points, current_point]

    return artists


ani = animation.FuncAnimation(
    fig, animate, frames=m * n * max_iter, interval=1, blit=False
)

ani.save("animation.mp4", fps=60, dpi=100, writer="ffmpeg")

# plt.show()
