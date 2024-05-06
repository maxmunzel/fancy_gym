import fancy_gym
import numpy as np
import gym
from gym import spaces
import sys
from fancy_gym.envs.mujoco.box_pushing.mp_wrapper import MPWrapper


def example_fully_custom_mp(seed=1, iterations=1, render=True):
    base_env_id = "BoxPushingDense-v0"
    wrappers = [MPWrapper]
    trajectory_generator_kwargs = {
        "trajectory_generator_type": "promp",
        "action_dim": 2,
        "weight_scale": 2,
    }
    phase_generator_kwargs = {"phase_generator_type": "linear"}
    controller_kwargs = {"controller_type": "velocity"}
    basis_generator_kwargs = {
        "basis_generator_type": "zero_rbf",
        "num_basis": 5,
        'num_basis_zero_start': 1,
    }
    env = fancy_gym.make_bb(
        env_id=base_env_id,
        wrappers=wrappers,
        black_box_kwargs={},
        traj_gen_kwargs=trajectory_generator_kwargs,
        controller_kwargs=controller_kwargs,
        phase_kwargs=phase_generator_kwargs,
        basis_kwargs=basis_generator_kwargs,
        seed=seed,
    )

    # # This is the magic line
    # for i in range(162):
    #     env = gym.make(f"ProDMP-BB-Random-Sweep-PosCtrl-{i}")
    for alpha in range(5,17,5):
        env = gym.make(f"Sweep78-alpha{alpha}-ws3-gs10")
        print("alpha ", alpha)
        env.env.env.traj_gen.show_scaled_basis(plot=True)
    #env = gym.make("Sweep53-tau5")

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
