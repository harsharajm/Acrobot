import math
from bisect import bisect_right

import gymnasium as gym
import numpy as np

N_BUCKETS = 8


def _edges(limit, n):
    # n buckets = n-1 inner edges, evenly spaced, with one edge exactly at 0.
    # The outer buckets are open-ended, so out-of-range values still land somewhere.
    return np.linspace(-limit, limit, n + 1)[1:-1].tolist()


EDGES = [
    _edges(math.pi, N_BUCKETS),  # angle 1 (rad)
    _edges(math.pi, N_BUCKETS),  # angle 2 (rad)
    _edges(6.0, N_BUCKETS),      # angular velocity 1: where the arm actually spends its time
    _edges(8.0, N_BUCKETS),      # angular velocity 2
]
STATE_SHAPE = tuple(len(e) + 1 for e in EDGES)  # (8, 8, 8, 8)


def make_env(render=False):
    return gym.make("Acrobot-v1", render_mode="human" if render else None)


def digitize_state(obs):
    # obs = [cos(th1), sin(th1), cos(th2), sin(th2), th1_dot, th2_dot]
    c1, s1, c2, s2, v1, v2 = obs.tolist()
    return (
        bisect_right(EDGES[0], math.atan2(s1, c1)),
        bisect_right(EDGES[1], math.atan2(s2, c2)),
        bisect_right(EDGES[2], v1),
        bisect_right(EDGES[3], v2),
    )


def reset(env):
    return digitize_state(env.reset()[0])


def step(env, action):
    obs, reward, terminated, truncated, _ = env.step(action)
    return digitize_state(obs), reward, terminated, truncated


def get_action_space(env):
    return int(env.action_space.n)