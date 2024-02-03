import os

import time
import json
import redis
import random
import numpy as np
from gym import utils, spaces
from copy import deepcopy
from gym.envs.mujoco import MujocoEnv
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
    rot_to_quat,
    get_quaternion_error,
    rotation_distance,
    affine_matrix_to_xpos_and_xquat,
)
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import (
    q_max,
    q_min,
    q_dot_max,
    q_torque_max,
)
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import desired_rod_quat
from typing import NamedTuple, Tuple, List
import random

import mujoco

MAX_EPISODE_STEPS_BOX_PUSHING = 100

BOX_POS_BOUND = np.array([[0.3, -0.45, -0.01], [0.6, 0.45, -0.01]])

if "REDIS_IP" in os.environ:
    redis_connection = redis.Redis(os.environ["REDIS_IP"], decode_responses=True)
else:
    redis_connection = None


Vec2 = Tuple[float, float]


class Trace(NamedTuple):
    goal_pos: Vec2
    finger_traj: List[Vec2]
    box_traj: List[Vec2]

    def save(self, plot_name: str = None):
        target_dir = os.environ.get("PLOT_DIR", None)
        if target_dir is None or not self.finger_traj:
            return
        plot_name = plot_name or str(random.randint(0, 99999999))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.scatter([self.goal_pos[0]], [self.goal_pos[1]], label="Goal")

        finger = np.array(self.finger_traj)
        box = np.array(self.box_traj)

        ax.plot(finger[:, 0], finger[:, 1], label="Finger")
        ax.plot(box[:, 0], box[:, 1], label="Box")
        fig.legend()
        fig.savefig(f"{target_dir}/{plot_name}.png")
        plt.close()


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

    def __init__(self, frame_skip: int = 10, random_init: bool = True):
        utils.EzPickle.__init__(**locals())
        self.trace = None
        self._steps = 0
        self.init_qpos_box_pushing = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.5,
                0.0,
                1.5,
                0.0,
                0.0,
                0.0,
                0.6,
                0.45,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            + [0.20, 0]
        )
        self.init_qvel_box_pushing = np.zeros(15 + 2)
        self.frame_skip = frame_skip

        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max
        self._desired_rod_quat = desired_rod_quat

        self._episode_energy = 0.0
        self.random_init = random_init
        self.session = random.randint(0, 99999999)
        MujocoEnv.__init__(
            self,
            model_path=os.path.join(
                os.path.dirname(__file__), "assets", "box_pushing.xml"
            ),
            frame_skip=self.frame_skip,
            mujoco_bindings="mujoco",
        )
        self.reset_model()

    def check_mocap(self):
        if redis_connection is not None:
            res = redis_connection.xrevrange("box_tracking", "+", "-", count=1)
            if res:
                _, payload = res[0]  # type: ignore
                transform = json.loads(payload["transform"])
                transform = np.array(transform).reshape(4, 4)
                box_pos, box_quat = affine_matrix_to_xpos_and_xquat(transform)
                i = mujoco.mj_name2id(self.model, 1, "mocap_box")
                self.model.body_pos[i] = box_pos.copy()
                self.model.body_quat[i] = box_quat.copy()
                # get rid of box
                self.data.body("box_0").xpos[2] = -0.5

    def push_to_redis(self):
        assert redis_connection is not None
        q = json.dumps(list(self.data.qpos.copy()))
        v = json.dumps(list(self.data.qvel.copy()))
        x, y, _ = self.data.body("finger").xpos.copy()
        payload = {
            "x": x,
            "y": y,
            "q": q,
            "v": v,
            "session": self.session,
            "cmd": "GOTO",
        }
        redis_connection.xadd("cart_cmd", payload)

    def step(self, action):
        # time.sleep(1 / 30)
        action = 1 * np.array(action).flatten()
        if self.trace is not None:
            # self.trace.finger_traj.append(tuple(action))
            self.trace.box_traj.append(tuple(self.data.body("finger").xpos[:2].copy()))
            self.trace.finger_traj.append(
                tuple(self.data.body("box_0").xpos[:2].copy())
            )

        desired_tcp_pos = self.data.body("finger").xpos.copy()
        desired_tcp_pos[2] += 0.055
        desired_tcp_quat = np.array([0, 1, 0, 0])

        q = self.data.qpos.copy()
        v = self.data.qvel.copy()
        self.desired_joint_pos = self.calculateOfflineIK(
            desired_tcp_pos, desired_tcp_quat
        )
        desired_joint_pos = self.desired_joint_pos
        self.data.qpos = q
        self.data.qvel = v

        self.data.qpos[:7] = desired_joint_pos
        self.data.qvel[:7] = 0

        unstable_simulation = False

        try:
            self.do_simulation(action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1
        self._episode_energy += np.sum(np.square(action))

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        box_pos = self.data.body("box_0").xpos.copy()
        box_quat = self.data.body("box_0").xquat.copy()
        target_pos = self.data.body("replan_target_pos").xpos.copy()
        target_quat = self.data.body("replan_target_pos").xquat.copy()
        rod_tip_pos = self.data.site("rod_tip").xpos.copy()
        rod_quat = self.data.body("push_rod").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        self.check_mocap()

        if not unstable_simulation:
            reward = self._get_reward(
                episode_end,
                box_pos,
                box_quat,
                target_pos,
                target_quat,
                rod_tip_pos,
                rod_quat,
                qpos,
                qvel,
                action,
            )
        else:
            reward = -50

        obs = self._get_obs()
        box_goal_pos_dist = (
            0.0 if not episode_end else np.linalg.norm(box_pos - target_pos)
        )
        box_goal_quat_dist = (
            0.0 if not episode_end else rotation_distance(box_quat, target_quat)
        )
        infos = {
            "episode_end": episode_end,
            "box_goal_pos_dist": box_goal_pos_dist,
            "box_goal_rot_dist": box_goal_quat_dist,
            "episode_energy": 0.0 if not episode_end else self._episode_energy,
            "is_success": True
            if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5
            else False,
            "num_steps": self._steps,
        }
        if redis_connection is not None:
            self.push_to_redis()
        print(f"Step complete, finger @ {self.data.body('finger').xpos[:2]}")

        return obs, reward, episode_end, infos

    def reset_model(self):
        if self.trace is not None:
            self.trace.save()
        # rest box to initial position
        self.set_state(self.init_qpos_box_pushing, self.init_qvel_box_pushing)
        box_init_pos = (
            self.sample_context()
            if self.random_init
            else np.array([0.4, 0.3, -0.01, 0.0, 0.0, 0.0, 1.0])
        )
        if redis_connection is not None:
            # get the box out of the way during real rollouts
            box_init_pos[2] = -0.5
            redis_connection.xadd("cart_cmd", {"cmd": "RESET"})
            print("Waiting for robot to reset")
            messages = redis_connection.xread(
                streams={"real_robot_obs": "$"}, count=1, block=0
            )
            assert messages
            message_id, payload = messages[0][1][-1]
            x = float(payload["x"])
            y = float(payload["y"])
            # self.data.body("finger").xpos[0] = x
            # self.data.body("finger").xpos[1] = y
            self.data.joint("finger_x_joint").qpos = x
            self.data.joint("finger_y_joint").qpos = y

            # self.data.qpos[mujoco.mj_name2id("finger_x_joint")] = x
            # self.data.qpos[mujoco.mj_name2id("finger_y_joint")] = y
            # self.data.joint("finger_x_joint").qpos = float(payload["x"])
            # self.data.joint("finger_y_joint").qpos = float(payload["y"])
            print("Waiting for mocap")
            self.check_mocap()
            print(f"Reset done, finger @ {x:.2f} {y:.2f}")

        else:
            self.data.joint("box_joint").qpos = box_init_pos
            self.data.joint("finger_x_joint").qpos = box_init_pos[0]
            self.data.joint("finger_y_joint").qpos = box_init_pos[1]

        # set target position
        box_target_pos = self.sample_context()
        while np.linalg.norm(box_target_pos[:2] - box_init_pos[:2]) < 0.3:
            box_target_pos = self.sample_context()

        # Derandomize
        self.data.body("replan_target_pos").xquat = box_target_pos = np.array(
            [0.51505285, 0.0, 0.0, 0.85715842]
        )
        self.data.body("replan_target_pos").xpos = np.array(
            [0.3186036 + 0.15, -0.25776725, -0.01]
        )

        self.trace = Trace(
            goal_pos=tuple(box_target_pos[:2]), finger_traj=[], box_traj=[]
        )
        # box_target_pos[0] = 0.4
        # box_target_pos[1] = -0.3
        # box_target_pos[-4:] = np.array([0.0, 0.0, 0.0, 1.0])
        self.model.body_pos[2] = box_target_pos[:3]
        self.model.body_quat[2] = box_target_pos[-4:]
        self.model.body_pos[3] = box_target_pos[:3]
        self.model.body_quat[3] = box_target_pos[-4:]

        # set the robot to the right configuration (rod tip in the box)
        desired_tcp_pos = box_init_pos[:3] + np.array([0.0, 0.0, 0.15])
        desired_tcp_quat = np.array([0, 1, 0, 0])
        desired_joint_pos = self.calculateOfflineIK(desired_tcp_pos, desired_tcp_quat)

        desired_joint_vel = desired_joint_pos - self.data.qpos[:7]

        self.data.qvel[:7] = desired_joint_vel
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.0

        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        theta = self.np_random.uniform(low=0, high=np.pi * 2)
        quat = rot_to_quat(theta, np.array([0, 0, 1]))
        return np.concatenate([pos, quat])

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        raise NotImplementedError

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos[:7].copy(),  # joint position
                self.data.qvel[:7].copy(),  # joint velocity
                # self.data.qfrc_bias[:7].copy(),  # joint gravity compensation
                # self.data.site("rod_tip").xpos.copy(),  # position of rod tip
                # self.data.body("push_rod").xquat.copy(),  # orientation of rod
                self.data.body("box_0").xpos.copy(),  # position of box
                self.data.body("box_0").xquat.copy(),  # orientation of box
                self.data.body("replan_target_pos").xpos.copy(),  # position of target
                self.data.body(
                    "replan_target_pos"
                ).xquat.copy(),  # orientation of target
            ]
        )
        return obs

    def _joint_limit_violate_penalty(
        self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False
    ):
        penalty = 0.0
        p_coeff = 1.0
        v_coeff = 1.0
        # q_limit
        if enable_pos_limit:
            higher_error = qpos - self._q_max
            lower_error = self._q_min - qpos
            penalty -= p_coeff * (
                abs(np.sum(higher_error[qpos > self._q_max]))
                + abs(np.sum(lower_error[qpos < self._q_min]))
            )
        # q_dot_limit
        if enable_vel_limit:
            q_dot_error = abs(qvel) - abs(self._q_dot_max)
            penalty -= v_coeff * abs(np.sum(q_dot_error[q_dot_error > 0.0]))
        return penalty

    def _get_box_vel(self):
        return self.data.body("box_0").cvel.copy()

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

    def calculateOfflineIK(self, desired_cart_pos, desired_cart_quat):
        """
        calculate offline inverse kinematics for franka pandas
        :param desired_cart_pos: desired cartesian position of tool center point
        :param desired_cart_quat: desired cartesian quaternion of tool center point
        :return: joint angles
        """
        old_q = self.data.qpos.copy()
        old_v = self.data.qvel.copy()
        J_reg = 1e-6
        w = np.diag([1, 1, 1, 1, 1, 1, 1])
        target_theta_null = np.array(
            [
                3.57795216e-09,
                1.74532920e-01,
                3.30500960e-08,
                -8.72664630e-01,
                -1.14096181e-07,
                1.22173047e00,
                7.85398126e-01,
            ]
        )
        eps = 1e-5  # threshold for convergence
        IT_MAX = 10
        dt = 1e-3
        i = 0
        pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        pgain_null = 5 * np.array(
            [
                7.675519770796831,
                2.676935478437176,
                8.539040163444975,
                1.270446361314313,
                8.87752182480855,
                2.186782233762969,
                4.414432577659688,
            ]
        )
        pgain_limit = 20
        q = self.data.qpos[:7].copy()
        qd_d = np.zeros(q.shape)
        old_err_norm = np.inf

        while True:
            q_old = q
            q = q + dt * qd_d
            q = np.clip(q, q_min, q_max)
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            current_cart_pos = self.data.body("tcp").xpos.copy()
            current_cart_quat = self.data.body("tcp").xquat.copy()

            cart_pos_error = np.clip(desired_cart_pos - current_cart_pos, -0.1, 0.1)

            if np.linalg.norm(current_cart_quat - desired_cart_quat) > np.linalg.norm(
                current_cart_quat + desired_cart_quat
            ):
                current_cart_quat = -current_cart_quat
            cart_quat_error = np.clip(
                get_quaternion_error(current_cart_quat, desired_cart_quat), -0.5, 0.5
            )

            err = np.hstack((cart_pos_error, cart_quat_error))
            err_norm = np.sum(cart_pos_error**2) + np.sum(
                (current_cart_quat - desired_cart_quat) ** 2
            )
            if err_norm > old_err_norm:
                q = q_old
                dt = 0.7 * dt
                continue
            else:
                dt = 1.025 * dt

            if err_norm < eps:
                break
            if i > IT_MAX:
                break

            old_err_norm = err_norm

            ### get Jacobian by mujoco
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)

            jacp = self.get_body_jacp("tcp")[:, :7].copy()
            jacr = self.get_body_jacr("tcp")[:, :7].copy()

            J = np.concatenate((jacp, jacr), axis=0)

            Jw = J.dot(w)

            # J * W * J.T + J_reg * I
            JwJ_reg = Jw.dot(J.T) + J_reg * np.eye(J.shape[0])

            # Null space velocity, points to home position
            qd_null = pgain_null * (target_theta_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (q_min + margin_to_limit - q)
            qd_null_limit[q > q_max - margin_to_limit] += qd_null_limit_max[
                q > q_max - margin_to_limit
            ]
            qd_null_limit[q < q_min + margin_to_limit] += qd_null_limit_min[
                q < q_min + margin_to_limit
            ]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        self.data.qpos = old_q
        self.data.qvel = old_v
        return q


class BoxPushingDense(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingDense, self).__init__(
            frame_skip=frame_skip, random_init=random_init
        )

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        joint_penalty = self._joint_limit_violate_penalty(
            qpos, qvel, enable_pos_limit=True, enable_vel_limit=True
        )
        tcp_box_dist_reward = -2 * np.clip(
            np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100
        )
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi
        energy_cost = -0.0005 * np.sum(np.square(action))

        reward = (
            joint_penalty
            + tcp_box_dist_reward
            + box_goal_pos_dist_reward
            + box_goal_rot_dist_reward
            + energy_cost
        )

        rod_inclined_angle = rotation_distance(rod_quat, self._desired_rod_quat)
        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        return reward


class BoxPushingTemporalSparse(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSparse, self).__init__(
            frame_skip=frame_skip, random_init=random_init
        )

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        reward = 0.0
        joint_penalty = self._joint_limit_violate_penalty(
            qpos, qvel, enable_pos_limit=True, enable_vel_limit=True
        )
        energy_cost = 0  # -0.02 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(
            np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100
        )
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
        box_goal_rot_dist_reward = (
            -rotation_distance(box_quat, target_quat) / np.pi * 100
        )

        ep_end_joint_vel = -50.0 * np.linalg.norm(qvel)

        reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward + ep_end_joint_vel

        return reward


class BoxPushingTemporalSpatialSparse(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSpatialSparse, self).__init__(
            frame_skip=frame_skip, random_init=random_init
        )

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        reward = 0.0
        joint_penalty = self._joint_limit_violate_penalty(
            qpos, qvel, enable_pos_limit=True, enable_vel_limit=True
        )
        energy_cost = -0.02 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(
            np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100
        )
        reward += joint_penalty + tcp_box_dist_reward + energy_cost
        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        if box_goal_dist < 0.1:
            reward += 300
            box_goal_pos_dist_reward = np.clip(-3.5 * box_goal_dist * 100 * 3, -100, 0)
            box_goal_rot_dist_reward = np.clip(
                -rotation_distance(box_quat, target_quat) / np.pi * 100 * 1.5, -100, 0
            )
            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward

        return reward


class BoxPushingTemporalSpatialSparse2(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingTemporalSpatialSparse2, self).__init__(
            frame_skip=frame_skip, random_init=random_init
        )

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        reward = 0.0
        joint_penalty = self._joint_limit_violate_penalty(
            qpos, qvel, enable_pos_limit=True, enable_vel_limit=True
        )
        energy_cost = -0.0005 * np.sum(np.square(action))
        tcp_box_dist_reward = -2 * np.clip(
            np.linalg.norm(box_pos - rod_tip_pos), 0.05, 100
        )

        reward += joint_penalty + energy_cost + tcp_box_dist_reward

        rod_inclined_angle = rotation_distance(rod_quat, desired_rod_quat)

        if rod_inclined_angle > np.pi / 4:
            reward -= rod_inclined_angle / (np.pi)

        if not episode_end:
            return reward

        # Force the robot to stop at the end
        reward += -50.0 * np.linalg.norm(qvel)

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        if box_goal_dist < 0.1:
            box_goal_pos_dist_reward = np.clip(-350.0 * box_goal_dist, -200, 0)
            box_goal_rot_dist_reward = np.clip(
                -rotation_distance(box_quat, target_quat) / np.pi * 100.0, -100, 0
            )
            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward
        else:
            reward -= 300.0

        return reward


class BoxPushingNoConstraintSparse(BoxPushingEnvBase):
    def __init__(self, frame_skip: int = 10, random_init: bool = False):
        super(BoxPushingNoConstraintSparse, self).__init__(
            frame_skip=frame_skip, random_init=random_init
        )

    def _get_reward(
        self,
        episode_end,
        box_pos,
        box_quat,
        target_pos,
        target_quat,
        rod_tip_pos,
        rod_quat,
        qpos,
        qvel,
        action,
    ):
        reward = 0.0
        joint_penalty = self._joint_limit_violate_penalty(
            qpos, qvel, enable_pos_limit=True, enable_vel_limit=True
        )
        energy_cost = -0.0005 * np.sum(np.square(action))
        reward += joint_penalty + energy_cost

        if not episode_end:
            return reward

        box_goal_dist = np.linalg.norm(box_pos - target_pos)

        box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100
        box_goal_rot_dist_reward = (
            -rotation_distance(box_quat, target_quat) / np.pi * 100
        )

        reward += (
            box_goal_pos_dist_reward
            + box_goal_rot_dist_reward
            + self._get_end_vel_penalty()
        )

        return reward

    def _get_end_vel_penalty(self):
        rot_coeff = 150.0
        pos_coeff = 150.0
        box_rot_pos_vel = self._get_box_vel()
        box_rot_vel = box_rot_pos_vel[:3]
        box_pos_vel = box_rot_pos_vel[3:]
        return -rot_coeff * np.linalg.norm(box_rot_vel) - pos_coeff * np.linalg.norm(
            box_pos_vel
        )
