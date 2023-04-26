import custom_impl_7
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from os.path import dirname, join, abspath

SCENE_FILE = join(dirname(abspath(__file__)), 'impl_7_scene.ttt')

env = gym.make("RobotEnv7-v0", scene_file=SCENE_FILE)
# It will check your custom environment and output additional warnings if needed
check_env(env)
