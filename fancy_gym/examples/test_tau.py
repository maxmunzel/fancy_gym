import gym
from fancy_gym.envs.mujoco.box_pushing.mp_wrapper import MPWrapper
from gym import register

i = 0
while True:
    name = f"Testing-{i}"
    tau = float(input("tau: ").strip())
    register(
        id=name,
        entry_point="fancy_gym.utils.make_env_helpers:make_bb_env_helper",
        kwargs={
            "name": "BoxPushingTemporalSparse-v0",
            "wrappers": [MPWrapper],
            "trajectory_generator_kwargs": {
                "trajectory_generator_type": "prodmp",
                "duration": 4.0,
                "action_dim": 2,
                "weight_scale": 0.1,
                "auto_scale_basis": True,
                "goal_scale": 0.3,
                "relative_goal": False,
                "disable_goal": False,
            },
            "phase_generator_kwargs": {
                "phase_generator_type": "exp",
                "tau": tau,
            },
            "controller_kwargs": {
                "controller_type": "position",
            },
            "basis_generator_kwargs": {
                "basis_generator_type": "prodmp",
                "alpha": 10,
                "num_basis": 3,
                # 'num_basis_zero_start': 1,
            },
            "random_init": True,
        },
    )
    env = gym.make(name)
    env.env.env.traj_gen.show_scaled_basis(plot=True)
    if input("Plot?").strip():
        for _ in range(5):
            env.reset()
            env.render(mode="human")
            ac = env.action_space.sample()
            env.step(ac)

