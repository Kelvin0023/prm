import numpy as np
import os

import matplotlib.pyplot as plt


class Box:
    def __init__(self, l, b, x, y):
        self.l = l
        self.b = b
        self.x = x
        self.y = y

    def isinside(self, xp, yp, pad=0.05):
        xbound = (xp > self.x - 0.5 * self.l - pad) and (
            xp < self.x + 0.5 * self.l + pad
        )
        ybound = (yp > self.y - 0.5 * self.b - pad) and (
            yp < self.y + 0.5 * self.b + pad
        )
        inside = xbound and ybound
        return inside


class Maze:
    def __init__(self, xrange=(-0.9, 0.9), yrange=(-0.9, 0.9)):
        self.obstacles = []
        self.xrange = xrange
        self.yrange = yrange

        self.areas = np.array([obs.area for obs in self.obstacles])

    def sample(self):
        while True:
            x = np.random.uniform(*self.xrange)
            y = np.random.uniform(*self.yrange)

            inside = False
            for obs in self.obstacles:
                if obs.isinside(x, y):
                    inside = True
                    break
            if not inside:
                return x, y


class MazeLvl0(Maze):
    def __init__(self, xrange=(-0.5, 0.5), yrange=(-0.5, 0.45)):
        super().__init__(xrange=xrange, yrange=yrange)
        self.obstacles = []


class MazeLvl1V2(Maze):
    def __init__(self, xrange=(-0.5, 0.9), yrange=(-0.5, 0.45)):
        super().__init__(xrange=xrange, yrange=yrange)
        self.obstacles = [
            Box(0.05, 0.8, -0.16666, -0.1),
            Box(0.05, 0.80, 0.16666, 0.1),
        ]


class MazeLvl2V2(Maze):
    def __init__(self, xrange=(-0.5, 0.9), yrange=(-0.5, 0.45)):
        super().__init__(xrange=xrange, yrange=yrange)
        self.obstacles = [
            Box(0.05, 0.8, -0.25, -0.1),
            Box(0.05, 0.80, 0.0, 0.0),
            Box(0.05, 0.80, 0.25, -0.1),
        ]

class MazeA(Maze):
    def __init__(self):
        super().__init__()

        self.obstacles = [
            Box(0.7, 0.1, 0.6, 0.3),
            Box(0.1, 0.7, 0.3, 0.6),
            Box(0.7, 0.1, -0.6, -0.3),
            Box(0.1, 0.7, -0.3, -0.6),
            Box(0.7, 0.1, 0.6, -0.3),
            Box(0.1, 0.7, 0.3, -0.6),
            Box(0.7, 0.1, -0.6, 0.3),
            Box(0.1, 0.7, -0.3, 0.6),
        ]

class MazeB(Maze):
    def __init__(self):
        super().__init__()

        self.obstacles = [
            Box(0.1, 0.8, -0.4, 0.0),
            Box(0.3, 0.1, -0.25, 0.4),
            Box(0.3, 0.1, -0.25, -0.4),
            Box(0.1, 0.8, 0.4, 0.0),
            Box(0.3, 0.1, 0.25, 0.4),
            Box(0.3, 0.1, 0.25, -0.4),
        ]


class MazeC(MazeB):
    def __init__(self):
        super().__init__()

        self.obstacles.extend(
            [
                Box(1.4, 0.1, 0.0, 0.7),
                Box(0.1, 0.5, -0.7, 0.45),
                Box(0.1, 0.5, 0.7, 0.45),
                Box(1.4, 0.1, 0.0, -0.7),
                Box(0.1, 0.5, -0.7, -0.45),
                Box(0.1, 0.5, 0.7, -0.45),
            ]
        )


def generate(maze, nsample=100):
    states = np.zeros((nsample, 4))

    for i in range(nsample):
        x, y = maze.sample()
        states[i, 0] = x
        states[i, 1] = y

    return states


def plot(states, pngfile):
    plt.plot(states[:, 0], states[:, 1], "b.")
    plt.savefig(pngfile)


if __name__ == "__main__":
    # maze = MazeLvl0()
    # name = "maze_lvl0"

    # maze = MazeLvl1V2()
    # name = "maze_lvl1_v2"

    # maze = MazeLvl2V2()
    # name = "maze_lvl2_v2"

    maze = MazeA()
    name = "maze_a"

    npfile = "tasks/maze/assets/reset_states/{}.npy".format(name)
    pngfile = "tasks/maze/assets/reset_states/{}.png".format(name)
    os.makedirs(os.path.dirname(npfile), exist_ok=True)
    state = generate(maze, nsample=10000)
    state = state.astype(dtype=np.float32)

    with open(npfile, "wb") as f:
        np.save(f, state)

    plot(state, pngfile)
