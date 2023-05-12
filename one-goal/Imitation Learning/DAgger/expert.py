from imitation.policies.base import HardCodedPolicy
import tempfile
import gymnasium as gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import custom_impl_8

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.logger import configure

class ExpertAgent(HardCodedPolicy):

    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
    
    def _choose_action(self, obs):

        agent_position = self.env.agent.get_position()
        agent_orientation = self.env.agent.get_orientation()
        goal_position = self.env.goal_pos
        goal_orientation = self.env.goal_orientation

        x_diff = goal_position[0] - agent_position[0]
        y_diff = goal_position[1] - agent_position[1]
        z_diff = goal_position[2] - agent_position[2]
        orientation_diff_z = goal_orientation[2] - agent_orientation[2]
        orientation_diff_z = min(orientation_diff_z, 2 * np.pi - orientation_diff_z)

        action = np.array([x_diff, y_diff, z_diff, orientation_diff_z], dtype=np.float64)
        # print(action)
        return action

if __name__ == "__main__":
    env = Monitor(gym.make("RobotEnv8-v0"), "logs/")
    venv = make_vec_env("RobotEnv8-v0", n_envs=16, vec_env_cls=SubprocVecEnv)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=np.random.default_rng(),
    )

    expert = ExpertAgent(env)

    tensorboard_log = "tensorboard_logs/"

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        logger = configure(folder=tensorboard_log, format_strs=["tensorboard", "stdout"])
        # logger.make_output_format("tensorboard")
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=np.random.default_rng(),
            custom_logger=logger,
        )

        dagger_trainer.train(2000)

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("reward", reward)
