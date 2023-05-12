from impl_6_code import RobotEnv6
from stable_baselines3.common.env_checker import check_env

env = RobotEnv6()
# It will check your custom environment and output additional warnings if needed
check_env(env)
