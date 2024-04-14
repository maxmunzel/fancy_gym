import fancy_gym
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from fancy_gym.envs.mujoco.box_pushing.mp_wrapper import MPWrapper


def example_fully_custom_mp(seed=1, iterations=1, render=True):
    env = gym.make("Sweep57-v0", render_mode="human")

    env.reset(seed=42)


    # number of samples/full trajectories (multiple environment steps)
    rewards = 0
    # env.env.env.traj_gen.show_scaled_basis(plot=True)
    for _ in range(iterations):
        # ac =  spaces.Box(low=np.array([ 0.15, -0.35]*6), high=np.array([0.55, 0.35]*6)).sample()

        ac = env.action_space.sample()
        _, reward, done, _, _ = env.step(ac)
        obs = env.reset()
        #env.env.env.traj_gen.show_scaled_basis(plot=True)
        #sys.exit(0)



if __name__ == "__main__":
    example_fully_custom_mp(seed=10, iterations=10000, render=True)
