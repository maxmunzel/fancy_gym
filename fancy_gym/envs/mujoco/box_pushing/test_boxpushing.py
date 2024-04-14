from .box_pushing_env import BoxPushingDense
import numpy as np
from pathlib import Path
import random
import json

render = False


def test_traj():
    random.seed(42)
    n_points = 8
    n_rep = 25  # at 50Hz of control this keeps every point active for .5 seconds
    reset_every = 60
    points = [
        (random.uniform(0.3, 0.6), random.uniform(-0.4, 0.4)) for _ in range(n_rep)
    ]

    refpath = Path(__file__).parent / "xpos_42.json"
    # env = BoxPushingDense()
    # env.reset(seed=42)

    # xpos = list()
    #
    # i = 0
    # for action in points:
    #     for _ in range(n_rep):
    #         i += 1
    #         env.step(action)
    #         if not i % reset_every:
    #             env.reset()
    #         if render:
    #             env.render("human")
    #         xpos.append(env.data.xpos.copy().ravel().tolist())

    # with open(refpath, "w") as f:
    #     json.dump(xpos, f, indent = 2)

    with open(refpath) as f:
        xpos = json.load(f)

    env = BoxPushingDense()
    env.reset(seed=42)

    i = 0
    for action in points:
        for _ in range(n_rep):
            ref_xpos = xpos[i]
            i += 1
            env.step(action)
            if not i % reset_every:
                env.reset()
            if render:
                env.render("human")
            assert np.allclose(
                ref_xpos, env.data.xpos.ravel().tolist()
            ), f"Xpos of BoxPushingEnv differs after {i} steps."


if __name__ == "__main__":
    test_traj()
