import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import transform
from scipy.spatial.transform import Rotation
from typing import Tuple
import redis
import json


def rotation_z_from_matrix(matrix):
    # Ensure the matrix is 4x4
    if matrix.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix")

    # Extract the rotation angle around the Z axis
    theta_radians = np.arctan2(matrix[1, 0], matrix[0, 0])
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees


def affine_matrix_to_xpos_and_xquat(matrix: np.ndarray):
    """
    takes a 4x4 matrix and returns a vector of the translation and quaternion as used by mujoco
    """
    assert matrix.shape == (4, 4)
    x, y, z = matrix[:3, 3].flatten()

    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to a quaternion
    qw = 0.5 * np.sqrt(
        1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    )
    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

    # Concatenate the translation and quaternion

    return np.array([x, y, z]), np.array([qw, qx, qy, qz])


import numpy as np


fig, ax = plt.subplots()  # pyright: ignore
fig: plt.Figure
ax: plt.Axes


x = 0


r = redis.Redis(decode_responses=True)


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

    box_w = 0.127
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


def animate(_):
    res = r.xrevrange("box_tracking", count=1)
    assert res
    _, payload = res[0]
    transform = np.array(json.loads(payload["transform"])).reshape(4, 4)

    print(f"{rotation_z_from_matrix(transform):3.2f}")

    box_pos, box_quat = affine_matrix_to_xpos_and_xquat(transform)
    box_pos = box_pos.flatten()[:2]
    box_quat = box_quat.flatten()

    ax.clear()
    ax.set_xlim(-0.37, 0.37)
    ax.set_ylim(0.65, 0.35)
    ax.set_aspect("equal")
    draw_pos(
        pos=box_pos,
        mujoco_quat=box_quat,
        ax=ax,
    )


ani = FuncAnimation(fig, animate)
plt.show()
