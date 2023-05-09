from imitation.policies.base import HardCodedPolicy
import tempfile
import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, make_vec_env

import custom_impl_8

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

class ExpertAgent(HardCodedPolicy):

    def __init__(self, env):
        super().__init__(env.observation_space, env.action_space)
        self.env = env
    
    def _choose_action(self, obs):
        print(self.env.agent.get_position())
        return np.array([0, 0, 0, 0])
       

env = gym.make("RobotEnv8-v0")
venv = make_vec_env("RobotEnv8-v0", n_envs=1, vec_env_cls=SubprocVecEnv)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=np.random.default_rng(),
)

expert = ExpertAgent(env)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=np.random.default_rng(),
    )

    dagger_trainer.train(2000)
