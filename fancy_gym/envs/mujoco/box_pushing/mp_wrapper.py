from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack(
            [
                [True] * 4,  # finger, box, target positions
                # [True] * 8,  # box, target orientation
                [True] * 2,  # sin/cos of box
            ]
        )

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.hstack(
            [
                self.data.joint("finger_x_joint").qpos,
                self.data.joint("finger_y_joint").qpos,
            ]
        )

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return np.hstack(
            [
                self.data.joint("finger_x_joint").qvel,
                self.data.joint("finger_y_joint").qvel,
            ]
        )
