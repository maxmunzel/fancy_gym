import fancy_gym
import numpy as np
import gym
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

    # This is the magic line
    env = gym.make("ProDMP-Motor-Paper-BB-Random")

    env.reset()

    if render:
        env.render(mode="human")

    # number of samples/full trajectories (multiple environment steps)
    rewards = 0
    for _ in range(iterations):
        ac = env.action_space.sample()
        _, reward, done, _ = env.step(ac)
        rewards += reward

        if done:
            print(rewards)
            rewards = 0
            env.reset()


if __name__ == "__main__":
    example_fully_custom_mp(seed=10, iterations=100, render=True)
