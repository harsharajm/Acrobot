import os
import pickle
import random

import numpy as np

import acrobot as ab

GAMMA = 0.99
ALPHA_START = 0.3  # learning rate decays linearly to ALPHA_END over training
ALPHA_END = 0.01
EPS_START = 0.1    # epsilon decays linearly to 0 over training


def create_q_table(env):
    return np.zeros(ab.STATE_SHAPE + (ab.get_action_space(env),))


def load_q_table(path):
    with open(path, "rb") as f:
        q = pickle.load(f)
    if q.shape[:-1] != ab.STATE_SHAPE:
        raise SystemExit(f"{path} was trained with different bins (shape {q.shape}). Retrain it.")
    return q


def save_q_table(q, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(q, f)
    print(f"Saved: {path}")


def run(env, q, episodes, learn, eps_start=0.0, alpha_start=ALPHA_START, alpha_end=ALPHA_END):
    n_actions = ab.get_action_space(env)
    rewards = []
    for ep in range(episodes):
        frac = ep / max(1, episodes - 1)
        epsilon = eps_start * (1 - frac)
        alpha = alpha_start + (alpha_end - alpha_start) * frac

        state = ab.reset(env)
        done = False
        total = 0.0
        while not done:
            row = q[state]  # view into the table, so updating row updates q
            if random.random() < epsilon:
                action = random.randrange(n_actions)
            else:
                action = int(row.argmax())
            next_state, reward, terminated, truncated = ab.step(env, action)
            done = terminated or truncated

            if learn:
                if terminated:  # goal reached: no future reward
                    target = reward
                else:           # a time-limit cutoff still bootstraps
                    target = reward + GAMMA * q[next_state].max()
                row[action] += alpha * (target - row[action])

            total += reward
            state = next_state

        rewards.append(total)
        if learn and (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}: avg of last 100 = {np.mean(rewards[-100:]):.1f}, "
                  f"epsilon = {epsilon:.2f}, alpha = {alpha:.3f}")
        elif not learn:
            print(f"Episode {ep + 1}: reward {total:.0f}")

    print(f"Average reward over {episodes} episodes: {np.mean(rewards):.1f}")
    return rewards
