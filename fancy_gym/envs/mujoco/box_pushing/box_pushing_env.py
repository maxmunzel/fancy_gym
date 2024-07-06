import os
import shutil
import tempfile
import json
import redis
import random
import numpy as np
from pathlib import Path
from gym import utils, spaces
from gym.envs.mujoco import MujocoEnv
from fancy_gym.envs.mujoco.box_pushing.throttle import Throttle
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
)
from fancy_gym.envs.mujoco.box_pushing.box_pushing_utils import desired_rod_quat
from typing import Tuple, Optional, Union
from doraemon import Doraemon, MultivariateBetaDistribution
import mujoco

MAX_EPISODE_STEPS_BOX_PUSHING = 300

BOX_POS_BOUND = np.array([[0.22, -0.35, -0.01], [0.58, 0.35, -0.01]])

if "REDIS_IP" in os.environ:
    redis_connection = redis.Redis(os.environ["REDIS_IP"], decode_responses=True)
else:
    redis_connection = None


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
        self.throttle = None
        self.doraemon = None
        utils.EzPickle.__init__(**locals())
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
        self.ee_speeds = []
        self.last_ee_pos = None

        self._q_max = q_max
        self._q_min = q_min
        self._q_dot_max = q_dot_max
        self._desired_rod_quat = desired_rod_quat

        self._episode_energy = 0.0
        self.random_init = random_init
        self.session = random.randint(0, 99999999)
        self.model_path = os.path.join(
            os.path.dirname(__file__), "assets", "box_pushing.xml"
        )
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            mujoco_bindings="mujoco",
        )
        # After the super messed it up
        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf])
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.2] * 14 + [-10, -10]),
            high=np.array([1.2] * 14 + [10, 10]),
            dtype=np.float64,
        )
        dist = MultivariateBetaDistribution(
            alphas=[1, 1, 1, 1, 1, 1],
            low=[-0.39, 0.30, 0, 0.20, 50, 0.3],
            high=[0.39, 0.67, 2 * np.pi, 0.20, 100, 0.3],
            param_bound=[1, 1, 1, 1, 1, 1],
            names=[
                "start_y",
                "start_x",
                "start_theta",
                "box_mass_factor",
                "kp",
                "friction",
            ],
            seed=42,
        )
        dist.random = self.np_random
        if redis_connection is not None:
            dist.set_params(np.ones_like(dist.get_params()))
        self.doraemon = Doraemon(
            dist=dist,
            k=200,
            kl_bound=0.2,
            target_success_rate=0.6,
        )
        # make it a little simpler for me to write the ymls
        # print("\n".join(f'"{k}",' for k in self.doraemon.param_dict().keys()))
        self.randomize()
        self.reset_model()
        print(f"Post-init random number: {self.np_random.integers(0, 999999)}")

    def randomize(self):
        self.sample, self.sample_dict = self.doraemon.dist.sample_dict()
        assets = Path(self.model_path).parent
        TMPDIR = os.environ.get("TMPDIR")

        with tempfile.TemporaryDirectory(
            prefix=TMPDIR + "/", suffix=str(random.randint(0, 9999999))
        ) as d:
            d = shutil.copytree(assets, f"{d}/assets")
            with open(f"{d}/finger.xml") as f_src:
                content = f_src.read()
            with open(f"{d}/finger.xml", "w") as f_dst:
                old = 'kp="200'
                assert old in content
                new = f'kp="{self.sample_dict["kp"]}'
                f_dst.write(content.replace(old, new))

            with open(f"{d}/push_box.xml") as f_src:
                content = f_src.read()
            with open(f"{d}/push_box.xml", "w") as f_dst:
                old = 'friction="0.4296'
                assert old in content
                new = f'friction="{self.sample_dict["friction"]}'
                f_dst.write(content.replace(old, new))

            self.model = self._mujoco_bindings.MjModel.from_xml_path(self.model_path)
            self.data = self._mujoco_bindings.MjData(self.model)

        if self.viewer:
            # update viewer, so rendering keeps working
            self.viewer.model = self.model
            self.viewer.data = self.data

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
                return box_pos, box_quat

    def push_to_redis(self, action):
        assert redis_connection is not None
        q = json.dumps(list(self.data.qpos.copy()))
        v = json.dumps(list(self.data.qvel.copy()))
        x, y, _ = self.data.body("finger").xpos.copy()
        payload = {
            "x": float(action[0]),
            "y": float(action[1]),
            "x_finger": x,
            "y_finger": y,
            "q": q,
            "v": v,
            "session": self.session,
            "cmd": "GOTO",
        }
        redis_connection.xadd("cart_cmd", payload)

    def step(self, action):
        action_clipped = np.clip(action, a_min=[0.35, -0.37], a_max=[0.65, 0.37])
        clipping_dist = np.linalg.norm(action - action_clipped)
        action = action_clipped
        if redis_connection is not None:
            if self.throttle is None:
                self.throttle = Throttle(target_hz=1 / self.dt, busy_wait=False)
            self.throttle.tick()
        # time.sleep(1 / 30)
        action = 1 * np.array(action).flatten()

        desired_tcp_pos = self.data.body("finger").xpos.copy()
        desired_tcp_pos[2] += 0.055

        q = self.data.qpos.copy()
        v = self.data.qvel.copy()
        self.data.qpos = q
        self.data.qvel = v

        unstable_simulation = False

        try:
            self.do_simulation(action, self.frame_skip)
        except Exception as e:
            print(e)
            unstable_simulation = True

        self._steps += 1

        episode_end = True if self._steps >= MAX_EPISODE_STEPS_BOX_PUSHING else False

        if redis_connection is not None:
            box_pos, box_quat = self.check_mocap()
        else:
            box_pos = self.data.body("box_0").xpos.copy()
            box_quat = self.data.body("box_0").xquat.copy()
        target_pos = self.data.body("replan_target_pos").xpos.copy()
        target_quat = self.data.body("replan_target_pos").xquat.copy()
        rod_tip_pos = self.data.body("finger").xpos.copy()
        rod_quat = self.data.body("finger").xquat.copy()
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()

        # Append to EE Speed History
        if self.last_ee_pos is None:
            self.last_ee_pos = rod_tip_pos[:2].copy()
        speed = np.linalg.norm(self.last_ee_pos - rod_tip_pos[:2]) / self.dt
        self.ee_speeds.append(speed)
        self.last_ee_pos = rod_tip_pos[:2].copy()

        # print(f"Speed: {speed:2.2f} Max Speed: {max(self.ee_speeds):3.2f}")

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

        reward -= clipping_dist
        too_fast = 0.0
        idle_time = np.mean(np.array(self.ee_speeds) <= 0.05)
        if episode_end:
            # Max EE Speed Panality -- ensure the trajectory is executable
            # Polymetis seems to only have joint speed limits but the following limit is based on the max ee speed
            # during the rollouts of Sweep70.
            speed_limit = 0.6  # m/s -- was .8
            max_speed = max(self.ee_speeds)
            #reward -= max_speed
            #if max_speed > speed_limit:
            #    too_fast = 1.0
            #    reward -= max_speed * 5
            #    reward -= 20
            #print(f"Max Speed: {max_speed:.2f}")
            #print(f"Idle time: {idle_time:.2f}")
            #print(f"Target_pos: ", target_pos)

            ## Also make sure we stop at the end of the episode
            # reward -= 10 * speed

        # calculate power cost
        self._episode_energy += speed**2

        obs = self._get_obs()

        box_goal_pos_dist = (
            0.0 if not episode_end else np.linalg.norm(box_pos - target_pos)
        )
        box_goal_quat_dist = (
            0.0 if not episode_end else rotation_distance(box_quat, target_quat)
        )
        is_success = (
            True
            if episode_end and box_goal_pos_dist < 0.05 and box_goal_quat_dist < 0.5
            else False
        )
        is_real_success = is_success and not too_fast

        if is_real_success:
            reward += 30

        if self.doraemon is None:
            infos = {}
        else:
            infos = {
                "episode_end": episode_end,
                "box_goal_pos_dist": box_goal_pos_dist,
                "box_goal_rot_dist": box_goal_quat_dist,
                "episode_energy": 0.0 if not episode_end else self._episode_energy,
                "is_success": is_success,
                "is_real_success": is_real_success,
                "num_steps": self._steps,
                "end_speed": speed,
                "too_fast": too_fast,
                "max_speed": max(self.ee_speeds),
                "idle_time": idle_time,
                "clipping_dist": clipping_dist,
            }
            infos.update(self.doraemon.param_dict())

        self.last_episode_successful = is_real_success

        if redis_connection is not None:
            self.push_to_redis(action)
            if episode_end:
                feedback = {k: float(v) for k, v in infos.items()}
                feedback["reward"] = reward
                feedback["is_success"] = int(
                    feedback["is_success"]
                )  # redis has no bools
                del feedback["episode_end"]
                redis_connection.xadd("episode_feedback", feedback)
        # print(
        #    f"Step complete, finger @ {self.data.body('finger').xpos[:2]}, reward={reward}"
        # )
        reward = np.nan_to_num(reward, nan=-300, posinf=-300, neginf=-300)

        return obs, reward, episode_end, infos

    def reset_model(self):
        if self.doraemon:
            self.doraemon.dist.random = self.np_random
        self.last_ee_pos = None
        self.ee_speeds = []
        self.throttle = None  # clear throttle so target time does not persist resets
        self.randomize()
        if self.doraemon is not None:
            self.doraemon.add_trajectory(self.sample, self.last_episode_successful)
            try:
                self.doraemon.update_dist()
            except Exception as e:
                print(f"update_dist() failed: {e}")

        # rest box to initial position
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
            box_err = self.np_random.uniform(low=-0.03, high=0.03, size=2)
            self.data.joint("box_rot_joint").qpos = self.sample_dict["start_theta"]
            self.data.joint("box_x_joint").qpos = box_init_pos[0] + box_err[0]
            self.data.joint("box_y_joint").qpos = box_init_pos[1] + box_err[1]
            self.data.joint("finger_x_joint").qpos = box_init_pos[0]
            self.data.joint("finger_y_joint").qpos = box_init_pos[1]

        # set target position
        box_target_pos = self.sample_context()
        # while np.linalg.norm(box_target_pos[:2] - box_init_pos[:2]) < 0.3:
        #    box_target_pos = self.sample_context()

        # Derandomize
        self.data.body("replan_target_pos").xquat = box_target_pos = np.array(
            [0.51505285, 0.0, 0.0, 0]
            # [0.51505285, 0.0, 0.0, 0.85715842]
        )
        self.data.body("replan_target_pos").xpos = np.array(
            # [0.3186036 + 0.15, -0.25776725, -0.01]
            [0.4, 0, -0.01]
        )

        # box_target_pos[0] = 0.4
        # box_target_pos[1] = -0.3
        # box_target_pos[-4:] = np.array([0.0, 0.0, 0.0, 1.0])
        self.model.body_pos[2] = box_target_pos[:3]
        self.model.body_quat[2] = box_target_pos[-4:]
        self.model.body_pos[3] = box_target_pos[:3]
        self.model.body_quat[3] = box_target_pos[-4:]

        # set the robot to the right configuration (rod tip in the box)
        box_init_pos[:3] + np.array([0.0, 0.0, 0.15])

        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        self._episode_energy = 0.0

        return self._get_obs()

    def sample_context(self):
        pos = self.np_random.uniform(low=BOX_POS_BOUND[0], high=BOX_POS_BOUND[1])
        pos[0] = self.sample_dict["start_x"]
        pos[1] = self.sample_dict["start_y"]
        theta = self.sample_dict["start_theta"]
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
        finger_pos = self.data.body("finger").xpos[:2].copy()
        if redis_connection is None:
            box_pos = self.data.body("box_0").xpos[:2].copy().flatten()
            box_quat = self.data.body("box_0").xquat.copy().flatten()
        else:
            box_pos = self.data.body("mocap_box").xpos[:2].copy().flatten()
            box_quat = self.data.body("mocap_box").xquat.copy().flatten()

        # Simulate measurement error. We assume perfect finger positions and box rotations.
        # Box positions are assumed to have additive noise.

        if redis_connection is not None:
            box_pos_measured = box_pos
        else:
            if self._steps == 0:
                # At the first step, the finger is put where we think the box is.
                # Therefore the agent always sees box_measured == finger.
                # We don't need to simulate measurement noise here,
                # as the reset_model() method will place the finger slightly off-center for us.
                box_pos_measured = finger_pos.copy()
            else:
                # In all other steps, take the simulated box and add noise
                box_err = self.np_random.uniform(low=-0.03, high=0.03, size=2)
                box_pos_measured = box_pos #  + box_err

        obs = np.concatenate(
            [
                finger_pos,
                box_pos_measured,
                self.data.body("replan_target_pos")
                .xpos[:2]
                .copy(),  # position of target
                box_quat,
                self.data.body(
                    "replan_target_pos"
                ).xquat.copy(),  # orientation of target
                self.data.body("finger").cvel[:2].copy(),
            ]
        )
        obs = np.nan_to_num(obs, nan=0)
        obs = np.clip(obs, -1, 1)

        if redis_connection is not None:
            redis_connection.xadd(
                "observation",
                {
                    "observation": json.dumps(obs.tolist()),
                },
            )
        return obs

    def _joint_limit_violate_penalty(
        self, qpos, qvel, enable_pos_limit=False, enable_vel_limit=False
    ):
        return 0
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
        box_goal_pos_dist_reward = -3.5 * np.linalg.norm(box_pos - target_pos)
        box_goal_rot_dist_reward = -rotation_distance(box_quat, target_quat) / np.pi

        reward = box_goal_pos_dist_reward + box_goal_rot_dist_reward
        if self.ee_speeds:
            # Max EE Speed Panality -- ensure the trajectory is executable
            # Polymetis seems to only have joint speed limits but the following limit is based on the max ee speed
            # during the rollouts of Sweep70.
            speed_limit = 0.6  # m/s -- was .8
            speed = self.ee_speeds[-1]
            reward -= .005 * speed
            # if speed > speed_limit:
            #     reward -= speed * 5
            #     reward -= 2

        return (reward * 100) / MAX_EPISODE_STEPS_BOX_PUSHING


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
        if episode_end:
            box_goal_dist = np.linalg.norm(box_pos - target_pos)
            box_goal_pos_dist_reward = -3.5 * box_goal_dist * 100

            box_goal_rot_dist_reward = (
                -rotation_distance(box_quat, target_quat) / np.pi * 100
            )

            box_goal_rot_dist_reward += (box_goal_rot_dist_reward / 20) ** 2

            reward += box_goal_pos_dist_reward + box_goal_rot_dist_reward
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
        energy_cost = 0  # -0.0005 * np.sum(np.square(action))
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
        energy_cost = 0  # -0.0005 * np.sum(np.square(action))
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
