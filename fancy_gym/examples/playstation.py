import pygame
import time
from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingDense
import fancy_gym

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Assume the PS4 controller is joystick 0
joystick.init()

# Setup OpenAI Gym environment
env = BoxPushingDense(random_init=True, frame_skip=20)
env = fancy_gym.make("BoxPushingDense-v0", seed=42)
obs = env.reset()

cum_reward = 0
while True:
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
    action = [y_axis, x_axis]

    print(action)

    # Step through the environment with the chosen action
    obs, reward, done, info = env.step(action)
    cum_reward *= 0.99
    cum_reward += reward

    # Render the environment
    env.render(mode="human")

    # Print the reward
    print(f"Reward: {cum_reward:3.2f} vel: {env.data.qpos[:7]}")
    # print(f"Err: {env.joint_controller.error_norm():4.2f} Int: {env.joint_controller.integral_norm():4.2f}")

    # If the episode is done, reset the environment
    # if done:
    #    obs = env.reset()
    # time.sleep(1/30)

pygame.quit()
