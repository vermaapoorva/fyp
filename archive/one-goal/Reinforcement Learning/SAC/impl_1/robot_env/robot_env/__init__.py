from gymnasium.envs.registration import register

register(id="RobotEnv-v0",
         entry_point="robot_env.envs:RobotEnv")