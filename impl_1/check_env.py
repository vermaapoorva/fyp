from impl_1_code import RobotEnv1 
from stable_baselines3.common.env_checker import check_env

env = RobotEnv1()
# It will check your custom environment and output additional warnings if needed
check_env(env)