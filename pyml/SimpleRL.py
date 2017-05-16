import numpy as np
import tensorflow as tf
from utils import *
import gym
import time

def guess_cartpole():
    env = gym.make('MountainCar-v0')
    w = np.zeros(env.observation_space.shape).astype('float32')
    print(w.shape)
    b = np.array(0).astype('float32')
    param = [w, b]
    def p(o):
        return np.sign(np.dot(o, w) + b) > 0

    def get_run(pw, pb, render=False):
        w, b = param
        w += pw
        b += pb
        o = env.reset()
        d = False
        R = 0
        while not d:
            if render:
                env.render()
                time.sleep(0.01)
            o, r, d, _ = env.step(p(o))
            R += r
        return R

    best_score = 0
    for i in range(1000):
        # get_run(0, 0, True)
        perts = [(np.random.normal(size=env.observation_space.shape).astype('float32'),
            np.random.normal()) for p in range(1000)]
        res = list(map(
            lambda p: (get_run(*p), p), perts))
        score, best = max(res, key=lambda v: v[0])
        if best_score < score:
            w += best[0]
            b += best[1]
            best_score = score
        print("Best score:", best_score)

if __name__ == '__main__':
    guess_cartpole()
