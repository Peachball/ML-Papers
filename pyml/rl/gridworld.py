import numpy as np
import gym
from gym import error, spaces, utils
import sys
from six import StringIO

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self):
        DIM = 4
        self.dim = DIM
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(DIM * DIM)
        self.player = (0, 0)
        self.grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 2]])
        self.transitions = {
                0: (0, -1), # LEFT
                1: (1, 0),  # DOWN
                2: (0, 1),  # RIGHT
                3: (-1, 0)  # UP
                }
        self._reset()


    def _step(self, action):
        t = self.transitions[action]
        x, y = self.player
        ns = (x + t[0], y + t[1])
        r = 0
        if 0 > ns[0] or ns[0] >= 4 or 0 > ns[1] or ns[1] >= 4: # No movement
            ns = self.player
            r = -1

        if self.grid[ns] == 2:
            r = 1
        if self.grid[ns] == 1:
            r = -1


        self.player = ns
        return (self._get_state(), r, r != 0, {})


    def _get_state(self):
        return self.player[0] * self.dim + self.player[1]


    def _reset(self):
        self.player = (0, 0)
        return self._get_state()


    def _render(self, mode='human', close=False):
        if close:
            return
        outgrid = self.grid
        pval = outgrid[self.player]
        outgrid[self.player] = 3
        out = StringIO() if mode == 'ansi' else sys.stdout
        out.write("\n".join(map(
            lambda r: "".join(map(lambda n: str(n), r)), outgrid)) +
            '\n')
        outgrid[self.player] = pval
        return out


if __name__ == '__main__':
    env = GridWorld()
    env.reset()
