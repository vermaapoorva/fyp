import gymnasium as gym
import robot_env
import time

def run_model():

    scene_file_name = "wooden_block_scene.ttt"
    bottleneck = [0.0843, -0.0254, 0.732, 1.100]
    env = gym.make("RobotEnv-v2", headless=False, image_size=64, sleep=0, file_name=scene_file_name, bottleneck=bottleneck)

    env.reset()
    env.draw_distribution()
    time.sleep(1000000)

    # while True:
    #     obs, _ = env.reset()
    #     done = False

    #     while not done:

    #         random_action = env.action_space.sample()

    #         obs, reward, done, truncated, info = env.step(random_action)

run_model()