from gymnasium.envs.registration import register

register(id="RobotEnvSAC-v0",
         entry_point="robot_env.envs:RobotEnv")