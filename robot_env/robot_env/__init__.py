from gymnasium.envs.registration import register

register(id="RobotEnv-v2",
         entry_point="robot_env.envs:RobotEnv")