import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from typing import Tuple

from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense

import numpy as np
import pygame

from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import (
    BoxPushingDense,
    BoxPushingTemporalSparse,
)

fig, ax = plt.subplots()  # pyright: ignore
fig: plt.Figure
ax: plt.Axes


x = 0


env = BoxPushingTemporalSparse(random_init=True, frame_skip=10)
# env = fancy_gym.make("BoxPushingDense-v0", seed=42)
obs = env.reset(seed=42)


def draw_pos(
    pos: np.ndarray,
    mujoco_quat: np.ndarray,
    ax: plt.Axes,
    color_body: str = "orange",
    color_face: str = "blue",
    alpha=1,
):
    pos = pos.flatten()
    assert pos.shape == (2,)

    mujoco_quat = mujoco_quat.flatten()
    assert mujoco_quat.shape == (4,)
    quat = mujoco_quat[[1, 2, 3, 0]]

    M = np.eye(4)
    M[:2, 3] = pos
    M[:3, :3] = Rotation.from_quat(quat).as_matrix()

    def proj2d(x: float, y: float) -> Tuple[float, float]:
        vec = M @ np.array([x, y, 0, 1])
        vec = vec.flatten().tolist()
        # swap x and y so the plot gets the perspective of the operator
        # standing in front of the table
        return vec[1], vec[0]

    def draw_line(p1: Tuple[float, float], p2: Tuple[float, float], color: str):
        x1, y1 = proj2d(*p1)
        x2, y2 = proj2d(*p2)
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha)

    box_w = 0.1
    d = box_w / 2
    #    (d, d)  │######│  (d, -d)      ▲ x
    #            │      │               │
    #            │      │               │     -y
    #   (-d, d)  └──────┘  (-d, -d)     └─────►
    # points going clockwise starting from the top right
    points = [(d, -d), (-d, -d), (-d, d), (d, d)]
    # draw body
    for p1, p2 in zip(points, points[1:]):
        draw_line(p1, p2, color=color_body)
    # draw face
    draw_line(points[0], points[-1], color=color_face)

finger_pos = np.zeros(2)
def animate(_):
    global finger_pos
    for event in pygame.event.get():
        # If you want to handle specific events, you can do that here
        if event.type == pygame.JOYBUTTONDOWN:
            if joystick.get_button(0):  # X button is typically indexed at 0
                obs = env.reset()
                cum_reward = 0
        pass

    x_axis = joystick.get_axis(0)  # Left stick horizontal
    y_axis = joystick.get_axis(1)  # Left stick vertical

    joy = np.array([y_axis, x_axis])

    obs, reward, done, info = env.step(finger_pos + .1 * joy)

    finger_pos = obs[:2]
    box_pos = obs[2:4]
    target_pos = obs[4:6]

    box_quat = obs[6:10]
    target_quat = obs[10:14]

    env.render(mode="human")
    ax.clear()
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(0.6, 0.2)
    ax.set_aspect("equal")
    ax.scatter([obs[1]], [obs[0]], label="Finger")
    draw_pos(
        pos=box_pos,
        mujoco_quat=box_quat,
        ax=ax,
    )
    draw_pos(
        pos=target_pos,
        mujoco_quat=target_quat,
        ax=ax,
        color_body="green",
        color_face="blue",
    )


# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Assume the PS4 controller is joystick 0
joystick.init()


try:
    ani = FuncAnimation(fig, animate)
    plt.show()
finally:
    pygame.quit()
