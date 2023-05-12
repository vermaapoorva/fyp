from impl_2_code import RobotEnv2
from stable_baselines3.common.env_checker import check_env

env = RobotEnv2()
# It will check your custom environment and output additional warnings if needed
check_env(env)