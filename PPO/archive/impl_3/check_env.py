from impl_3_code_copy2 import RobotEnv3
from stable_baselines3.common.env_checker import check_env

env = RobotEnv3()
# It will check your custom environment and output additional warnings if needed
check_env(env)