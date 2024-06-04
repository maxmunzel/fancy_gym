from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack(
            [
                [True] * 6,  # finger, box, target positions
                [True] * 8,  # box, target orientation
            ]
        )

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return (
            self.env.sim2obs
            @ np.hstack(
                [
                    self.data.joint("finger_x_joint").qpos,
                    self.data.joint("finger_y_joint").qpos,
                    [0],
                    [1],
                ]
            )
        ).flatten()[:2]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return (
            self.env.sim2obs
            @ np.hstack(
                [
                    self.data.joint("finger_x_joint").qvel,
                    self.data.joint("finger_y_joint").qvel,
                    [0],
                    [0],
                ]
            )
        ).flatten()[:2]
