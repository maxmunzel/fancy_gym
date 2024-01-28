import pygame
import numpy as np
import time
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense
import fancy_gym
import os

FPS = 40
dt = BoxPushingDense().dt
skip = int(0.9 * (1 / FPS) / dt)
skip = max(1, skip)
print(skip)

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Assume the PS4 controller is joystick 0
joystick.init()

# Setup OpenAI Gym environment
env = BoxPushingDense(random_init=False, frame_skip=skip)
# env = fancy_gym.make("BoxPushingDense-v0", seed=42)
obs = env.reset()

cum_reward = 0
start_time = time.time()
frame = 0
while True:
    frame += 1
    for event in pygame.event.get():
        # If you want to handle specific events, you can do that here
        if event.type == pygame.JOYBUTTONDOWN:
            if joystick.get_button(0):  # X button is typically indexed at 0
                obs = env.reset()
                cum_reward = 0
        pass

    # Read joystick axes for actions
    x_axis = joystick.get_axis(0)  # Left stick horizontal
    y_axis = joystick.get_axis(1)  # Left stick vertical

    # Here, I assume the environment expects a 2D action.
    # Adjust this as per the action space of your environment.
    action = np.array([y_axis, x_axis])
    action *= 1

    # print(action)

    # Step through the environment with the chosen action
    obs, reward, done, info = env.step(action)
    cum_reward *= 0.99
    cum_reward += reward

    # Render the environment
    if "REDIS_IP" in os.environ:
        if not frame % 20:
            env.render(mode="human")
    else:
        if not frame % 4:
            env.render(mode="human")

    # print(f"Reward: {cum_reward:3.2f} vel: {env.data.qpos[:7]}")

    # keep simulation running in approximate real time
    t_real = time.time() - start_time
    t_sim = frame * dt
    error = t_sim - t_real

    if error > 0:
        pass
        # time.sleep(error)
    if "REDIS_IP" in os.environ and abs(error) > 10:
        # we don't want the simulation to catch up to the real time,
        # because this could cause exessively fast movements
        print("Error, simulation lagging behind!")
        break

    print(error)

pygame.quit()
