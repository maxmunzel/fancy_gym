import time
from typing import Dict
import numpy as np
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense
import fancy_gym
import redis
import json


def main(redis_ip: str = "localhost"):
    env = BoxPushingDense(random_init=True, frame_skip=20)
    env = fancy_gym.make("BoxPushingDense-v0", seed=42)
    env.reset()
    action = [0, 0]
    while True:
        env.step(action)
        env.render(mode="human")
        time.sleep(1 / 30)

    while True:
        res = r.xrevrange("cart_cmd", "+", "-", count=1)
        if res:
            # res = [('1704280318147-0', {...})]
            payload: Dict = res[0][1]
            x = np.array(json.loads(payload[b"x"]))
            y = np.array(json.loads(payload[b"y"]))

            env.step([x,y])

        # Render the environment
        env.render(mode="human")
        time.sleep(1 / 30)


if __name__ == "__main__":
    import typer

    typer.run(main)
