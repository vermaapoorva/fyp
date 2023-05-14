import custom_impl_8
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from os.path import dirname, join, abspath

# SCENE_FILE = join(dirname(abspath(__file__)), 'impl_8_scene.ttt')

env = gym.make("RobotEnv8-v0")
# It will check your custom environment and output additional warnings if needed
check_env(env)
