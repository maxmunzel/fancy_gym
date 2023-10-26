import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import rot_to_quat, get_quaternion_error, rotation_distance
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import q_max, q_min, q_dot_max, q_torque_max
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import desired_rod_quat

import mujoco

MAX_EPISODE_STEPS_BOX_PUSHING = 100

BOX_POS_BOUND = np.array([[0.3, -0.45, -0.01], [0.6, 0.45, -0.01]])

class BoxPushingEnvBase(MujocoEnv, utils.EzPickle):
    """
    franka box pushing environment
    action space:
        normalized joints torque * 7 , range [-1, 1]
    observation space:

    rewards:
    1. dense reward
    2. time-depend sparse reward
    3. time-spatial-depend sparse reward
    """

    _target_pos = np.zeros(3)
    _target_quat = np.zeros(4)

    def __init__(self, frame_skip: int = 10, random_init: bool = True):
        utils.EzPickle.__init__(**locals())
        self._steps = 0
        self.frame_skip = frame_skip

        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max
        self._desired_rod_quat = desired_rod_quat

        self._episode_energy = 0.
        self.random_init = random_init
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), "assets", "2d_box_pushing.xml"),
                           frame_skip=self.frame_skip,
                           mujoco_bindings="mujoco")
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def step(self, action):
        action = 10 * np.clip(action, self.action_space.low, self.action_space.high)

        unstable_simulation = False

        try:
            self.do_simulation(action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        box_pos = self.get_box_pos()
        box_quat = self.get_box_quat()
        target_pos = self.get_target_pos()
        target_quat = self.get_target_quat()
        rod_tip_pos = self.get_finger_pos()

        if not unstable_simulation:
            reward = self._get_reward(episode_end, box_pos, box_quat, target_pos, target_quat,
                                      rod_tip_pos, action)
        else:
            reward = -50

        obs = self._get_obs()
        box_goal_pos_dist = 0. if not episode_end else np.linalg.norm(box_pos - target_pos)
        box_goal_quat_dist = 0. if not episode_end else rotation_distance(box_quat, target_quat)
        infos = {
            'episode_end': episode_end,
            'box_goal_pos_dist': box_goal_pos_dist,
            'box_goal_rot_dist': box_goal_quat_dist,
            'episode_energy': 0. if not episode_end else self._episode_energy,
            'is_success': True if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5 else False,
            'num_steps': self._steps
        }
        return obs, reward, episode_end, infos

    def reset_model(self):
        box_limit_x = (.2, 0.6)
        box_limit_y = (-0.8, 0.8)

        # draw box pos
        x = np.random.uniform(*box_limit_x)
        y = np.random.uniform(*box_limit_y)


        self.set_box_pos(x, y)
        self.set_finger_pos(x, y)

        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        self.set_box_rotation(theta)

        target_x = x
        target_y = y
        while np.linalg.norm(np.array([target_x, target_y]) - [x,y]) < 0.3:
            target_x = np.random.uniform(*box_limit_x)
            target_y = np.random.uniform(*box_limit_y)

        target_theta = theta # self.np_random.uniform(low=0, high=np.pi * 2)
        self.set_target_pos_and_rotation(target_x, target_y, target_theta)


        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.

        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        return np.concatenate([pos, quat])

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        raise NotImplementedError

    def _get_obs(self):
        obs = np.concatenate([
            # self.data.qpos[:7].copy(),  # joint position
            # self.data.qvel[:7].copy(),  # joint velocity
            # self.data.qfrc_bias[:7].copy(),  # joint gravity compensation
            # self.data.site("rod_tip").xpos.copy(),  # position of rod tip
            # self.data.body("push_rod").xquat.copy(),  # orientation of rod
            self.get_finger_pos(),
            self.get_box_pos(),  
            self.get_box_quat(),  
            self.get_target_pos(),  
            self.get_target_quat(),  
        ])
        return obs

    def _joint_limit_violate_penalty(self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False):
        penalty = 0.
        p_coeff = 1.
        v_coeff = 1.
        # q_limit
        if enable_pos_limit:
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (abs(np.sum(higher_error[qpos > self._q_max])) +
                                  abs(np.sum(lower_error[qpos < self._q_min])))
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_error > 0.]))
        return penalty


    def get_body_jacp(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, None, id)
        return jacp

    def get_body_jacr(self, name):
        id = mujoco.mj_name2id(self.model, 1, name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, None, jacr, id)
        return jacr

    def _get_box_vel(self):
        return self.data.body("box_0").cvel.copy()

    def get_box_pos(self):
        return self.data.body("box_0").xpos.copy() 

    def get_box_quat(self):
        return self.data.body("box_0").xquat.copy()

    def get_target_pos(self):
        return self._target_pos

    def get_target_quat(self):
        return self._target_quat

    def get_finger_pos(self):
        return self.data.site("finger_tip").xpos.copy()

    def set_box_pos(self, x_offset, y_offset):
        qpos = self.data.qpos.copy()
        qpos[0] = x_offset 
        qpos[1] = y_offset 
        self.data.qpos = qpos

    def set_finger_pos(self, x_offset, y_offset):
        qpos = self.data.qpos.copy()
        qpos[7] = x_offset 
        qpos[8] = y_offset 
        self.data.qpos = qpos

    def set_box_rotation(self, theta):
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        qpos = self.data.qpos.copy()
        qpos[3:7] = quat
        self.data.qpos = qpos

    def set_target_pos_and_rotation(self, x, y, theta):
        pos = np.array([x,y,0])
        quat = rot_to_quat(theta, np.array([0, 0, 1]))

        self.model.body_pos[3] = pos[:3].copy() 
        self.model.body_quat[3] = quat.copy()

        self._target_quat = quat
        self._target_pos = pos


class BoxPushingDense(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingDense, self).__init__(frame_skip=frame_skip, random_init=random_init)
    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, action):
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        energy_cost = -0.0005 * np.sum(np.square(action))

        reward = tcp_box_dist_reward + \
                 box_goal_pos_dist_reward + box_goal_rot_dist_reward + energy_cost


        return reward

class BoxPushingTemporalSparse(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSparse, self).__init__(frame_skip=frame_skip, random_init=random_init)

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.02 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi * 100

        ep_end_joint_vel = -50. * np.linalg.norm(qvel)

        reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward + ep_end_joint_vel

        return reward


class BoxPushingTemporalSpatialSparse(BoxPushingEnvBase):

    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSpatialSparse, self).__init__(frame_skip=frame_skip, random_init=random_init)

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.02 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        if box_goal_dist < 0.1:
            reward += 300
            box_goal_pos_dist_reward = np.clip(- 3.5 * box_goal_dist * 100 * 3, -100, 0)
            box_goal_rot_dist_reward = np.clip(- rotation_distance(box_quat, target_quat)/np.pi * 100 * 1.5, -100, 0)
            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward

        return reward

class BoxPushingTemporalSpatialSparse2(BoxPushingEnvBase):

    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSpatialSparse2, self).__init__(frame_skip=frame_skip, random_init=random_init)

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.0005 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100)

        reward += joint_penalty + energy_cost + tcp_box_dist_reward

        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        # Force the robot to stop at the end
        reward += -50. * np.linalg.norm(qvel)

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        if box_goal_dist < 0.1:
            box_goal_pos_dist_reward = np.clip(- 350. * box_goal_dist, -200, 0)
            box_goal_rot_dist_reward = np.clip(- rotation_distance(box_quat, target_quat)/np.pi * 100., -100, 0)
            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward
        else:
            reward -= 300.

        return reward


class BoxPushingNoConstraintSparse(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingNoConstraintSparse, self).__init__(frame_skip=frame_skip, random_init=random_init)

    def _get_reward(self, episode_end, box_pos, box_quat, target_pos, target_quat,
                    rod_tip_pos, rod_quat, qpos, qvel, action):
        reward = 0.
        joint_penalty = self._joint_limit_violate_penalty(qpos, qvel, enable_pos_limit=True, enable_vel_limit=True)
        energy_cost = -0.0005 * np.sum(np.square(action))
        reward += joint_penalty + energy_cost

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi * 100

        reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward + self._get_end_vel_penalty()

        return reward

    def _get_end_vel_penalty(self):
        rot_coeff = 150.
        pos_coeff = 150.
        box_rot_pos_vel = self._get_box_vel()
        box_rot_vel = box_rot_pos_vel[:3]
        box_pos_vel = box_rot_pos_vel[3:]
        return -rot_coeff * np.linalg.norm(box_rot_vel) - pos_coeff * np.linalg.norm(box_pos_vel)
