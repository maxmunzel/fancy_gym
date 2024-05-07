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

    # # This is the magic line
    # for i in range(162):
    #     env = gym.make(f"ProDMP-BB-Random-Sweep-PosCtrl-{i}")
    # Store all idle times
    # idle_times_by_alpha = defaultdict(list)
    for alpha in range(5,30, 10):
        print("alpha ",alpha)
        env = gym.make(f"Sweep68-alpha{alpha}-tau5-ws10-gs10")
        env.env.env.traj_gen.show_scaled_basis(plot=True)

    for alpha in range(5, 30):
        #env = gym.make(f"Sweep77-alpha{alpha}-tau5-ws3-gs10")
        #env = gym.make(f"Sweep78-alpha{alpha}-ws3-gs10")
        env = gym.make(f"Sweep78-alpha{alpha}-ws10-gs10")
        print("alpha ", alpha)
        ac = env.action_space.sample()
        N = 10
        for _ in range(N):
            env.reset(seed=42)
            _, reward, done, info = env.step(ac)
            idle_times_by_alpha[alpha].append(info["idle_time"][-1])

    # Compute mean and standard error of the mean (SEM)
    x = list(idle_times_by_alpha.keys())
    means = [np.mean(idle_times_by_alpha[a]) for a in x]
    sems = [np.std(idle_times_by_alpha[a]) / np.sqrt(len(idle_times_by_alpha[a])) for a in x]

    # Plotting
    plt.errorbar(x, means, yerr=sems, fmt='-o')  # 'fmt' is the format of the line and points
    plt.ylim(0, 1)
    plt.xlabel('Alpha')
    plt.ylabel('Average Idle Time')
    plt.title('Average Idle Time by Alpha with Confidence Intervals')
    plt.show()
    print(json.dumps(idle_times_by_alpha, indent=2))
    #env = gym.make("Sweep53-tau5")

    sys.exit(0)

    env.reset()

    if render:
        env.render(mode="human")

    # number of samples/full trajectories (multiple environment steps)
    rewards = 0
    env.env.env.traj_gen.show_scaled_basis(plot=True)
    for _ in range(iterations):
        # ac =  spaces.Box(low=np.array([ 0.15, -0.35]*6), high=np.array([0.55, 0.35]*6)).sample()

        ac = env.action_space.sample()
        _, reward, done, _ = env.step(ac)
        env.reset()
        #env.env.env.traj_gen.show_scaled_basis(plot=True)
        #sys.exit(0)
        rewards += reward



if __name__ == "__main__":
    example_fully_custom_mp(seed=10, iterations=100, render=True)
