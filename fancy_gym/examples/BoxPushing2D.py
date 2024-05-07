import fancy_gym
import numpy as np
import gym
from gym import spaces
import sys
from fancy_gym.envs.mujoco.box_pushing.mp_wrapper import MPWrapper
import json
from collections import defaultdict
import matplotlib.pyplot as plt


def example_fully_custom_mp(seed=1, iterations=1, render=True):
    env = gym.make(f"Sweep68-alpha15-tau5-ws10-gs10")
    env.reset()
    print(env.action_space)

    if render:
        env.render(mode="human")

    # number of samples/full trajectories (multiple environment steps)
    rewards = 0
    # env.env.env.traj_gen.show_scaled_basis(plot=True)
    for _ in range(iterations):
        ac = env.action_space.sample()
        _, reward, done, _ = env.step(ac)
        env.reset()
        rewards += reward



if __name__ == "__main__":
    example_fully_custom_mp(seed=10, iterations=100, render=True)
