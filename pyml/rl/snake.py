import numpy as np
import gym
from gym import error, spaces, utils
import random
import sys
from six import StringIO
import time

class Snake(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, size=10):
        self.board = np.zeros((size, size))
        self.size = size
        self.player = [self._gen_random_position()]
        self.board[self.player] = 1
        self._generate_apple()
        self.transitions = {
                0: [ 0  , -1 ]     , # LEFT
                1: [ 1  , 0  ]     , # DOWN
                2: [ 0  , 1  ]     , # RIGHT
                3: [ -1 , 0 ]        # UP
                }

        self.observation_space = spaces.Box(0, 1, shape=(size, size, 2))
        self.action_space = spaces.Discrete(4)

    def _generate_apple(self):
        self.apple = self._gen_random_position()
        self.board[tuple(self.apple)] = -1

    def _gen_random_position(self):
        cap = np.equal(self.board, 0).sum()
        p = random.randint(0, cap - 1)
        pos = [0, 0]
        for i in range(p + 1):
            while self.board[tuple(pos)] != 0:
                pos[1] += 1
                if pos[1] >= self.size:
                    pos[1] -= self.size
                    pos[0] += 1
            pos[1] += i != 0
            if pos[1] >= self.size:
                pos[1] -= self.size
                pos[0] += 1
        return pos

    def _add(self, x, y):
        return list(map(lambda v: v[0] + v[1], zip(x, y)))

    def _get_state(self):
        o = np.zeros((self.size, self.size, 2))
        o[self.player, 0] = 1
        o[self.apple, 1] = 1
        return o

    def _step(self, action):
        assert self.action_space.contains(action)
        t = self.transitions[action]
        r = 0
        terminal = False
        self.player = [self._add(t, self.player[0])] + self.player
        newpos = tuple(self.player[0])
        if not 0 <= newpos[0] < self.size or not 0 <= newpos[1] < self.size:
            r = -1
            terminal = True
        elif self.board[newpos] > 0:
            r = -1
            terminal = True
        elif self.board[newpos] < 0:
            r = 1
            self.board[newpos] = 1
            if len(self.player) == self.board.size:
                terminal = True
            else:
                self._generate_apple()
        else:
            self.board[newpos] = 1
            self.board[tuple(self.player[-1])] = 0
            self.player = self.player[:-1]

        return (self._get_state, r, terminal, {})

    def _reset(self):
        self.board[:,:] = 0
        self.player = [self._gen_random_position()]
        self.board[tuple(self.player[0])] = 1
        self._generate_apple()

    def _get_char(self, i):
        if i == 0:
            return '.'
        if i > 0:
            return 'X'
        if i < 0:
            return 'A'

    def _render(self, mode="human", close=False):
        if close:
            return
        out = StringIO() if mode == 'ansi' else sys.stdout
        out.write('\n'.join(map(
            lambda r: "".join(map(
                lambda n: self._get_char(n), r)), self.board)) + '\n')
        return out


def playSnake():
    env = Snake()
    t = False
    env.reset()
    char_mapping = {
            'a': 0,
            's': 1,
            'd': 2,
            'w': 3
            }
    pmove = 'a'
    reward = 0
    while not t:
        env.render()
        c = input()
        if c not in char_mapping:
            _, r, t, _ = env.step(char_mapping[pmove])
            reward += r
        else:
            pmove = c
            _, r, t, _ = env.step(char_mapping[c])
            reward += r
    print("Total score: {}".format(reward))


if __name__ == '__main__':
    playSnake()
