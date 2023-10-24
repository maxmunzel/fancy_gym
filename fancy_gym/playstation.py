import pygame
import time
from envs.mujoco.box_pushing.box_pushing_env_2d import BoxPushingDense

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)  # Assume the PS4 controller is joystick 0
joystick.init()

# Setup OpenAI Gym environment
env = BoxPushingDense()
obs = env.reset()

while True:
    for event in pygame.event.get():
        # If you want to handle specific events, you can do that here
        if event.type == pygame.JOYBUTTONDOWN:
            if joystick.get_button(0):  # X button is typically indexed at 0
                obs = env.reset()
        pass

    # Read joystick axes for actions
    x_axis = joystick.get_axis(0)  # Left stick horizontal
    y_axis = joystick.get_axis(1)  # Left stick vertical

    # Here, I assume the environment expects a 2D action.
    # Adjust this as per the action space of your environment.
    action = [y_axis, x_axis]

    # Step through the environment with the chosen action
    obs, reward, done, info = env.step(action)

    # Render the environment
    env.render(mode="human")

    # Print the reward
    print(f"Reward: {reward}")

    # If the episode is done, reset the environment
    #if done:
    #    obs = env.reset()
    time.sleep(1/30)

pygame.quit()
