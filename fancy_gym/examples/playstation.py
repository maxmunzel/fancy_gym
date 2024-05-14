import pygame
import numpy as np
import time
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import (
    BoxPushingDense,
    BoxPushingTemporalSparse,
)
from fancy_gym.envs.mujoco.box_pushing.throttle import Throttle
import fancy_gym
import os

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Assume the PS4 controller is joystick 0
joystick.init()

# Setup OpenAI Gym environment
env = BoxPushingTemporalSparse(random_init=True, frame_skip=10)
# env = fancy_gym.make("BoxPushingDense-v0", seed=42)
obs = env.reset(seed=42)

throttle = Throttle(target_hz=50, busy_wait=False)
try:
    while True:
        throttle.tick()
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

        finger_pos = env.data.body("finger").xpos.copy()[:2]
        action = finger_pos + 0.05 * joy
        action *= 1
        obs, reward, done, info = env.step(action)

        env.render(mode="human")
finally:
    pygame.quit()
