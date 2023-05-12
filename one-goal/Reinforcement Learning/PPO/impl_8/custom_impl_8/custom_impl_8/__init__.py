from gymnasium.envs.registration import register

register(id="RobotEnv8-v0",
         entry_point="custom_impl_8.envs:RobotEnv8")