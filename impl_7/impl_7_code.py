import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import custom_impl_7

from torchvision import models
import numpy as np
from PIL import Image
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Camera

import time
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

# os.environ['http_proxy']=''

SCENE_FILE = join(dirname(abspath(__file__)), 'impl_7_scene.ttt')

#################################################################################################
##########################################   CALLBACK   #########################################
#################################################################################################

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param model_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, logdir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.logdir = logdir
        self.save_path = os.path.join(logdir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.logdir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    
# Define a custom callback function to calculate the average reward per time step
class AvgRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.avg_reward = 0
        self.results = []

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            result = 0
            if self.locals.get("rewards")[0] == 200:
                result = 1
            tensorboard_callback.writer.add_scalar('Final reward', result, self.num_timesteps)
            self.results.append(result)
        return True

    def _on_training_end(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        print("rollout ends after {} episodes".format(len(self.results)))
        tensorboard_callback.writer.add_scalar('Average final reward', np.mean(self.results), self.num_timesteps)
        self.results = []

#################################################################################################
###################################   USING THE ENVIRONMENT   ###################################
#################################################################################################

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

iter = 17
logdir = "logs" + str(iter)
tensorboard_log_dir = "tensorboard_logs"
tensorboard_callback = TensorBoardOutputFormat(tensorboard_log_dir + "/Average final reward_" + str(iter))

def train():

    env = make_vec_env("RobotEnv7-v0", n_envs=5, vec_env_cls=SubprocVecEnv, monitor_dir=logdir, env_kwargs=dict(image_size=64))
    # env = gym.make("RobotEnv7-v0", scene_file=SCENE_FILE)
    # env = Monitor(env, logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    policy_kwargs = dict(
        net_arch = dict(pi=[128, 256, 256, 128], vf=[128, 256, 256, 128])
    )

    model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, batch_size=2048, verbose=1, tensorboard_log=tensorboard_log_dir)

    # Create the callbacks
    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, logdir=logdir)
    avg_reward_callback = AvgRewardCallback()

    # Train the agent
    timesteps = 500000000
    model.learn(total_timesteps=int(timesteps), callback=[save_best_model_callback, avg_reward_callback])
    plot_results([logdir], timesteps, results_plotter.X_TIMESTEPS, "PPO")
    plt.show()

def run_model():

    env = Monitor(gym.make("RobotEnv7-v0", headless=True, image_size=64, sleep=0), logdir)

    model_path = f"{logdir}/best_model.zip"
    model = PPO.load(model_path, env=env)

    total_episodes = 0
    successful_episodes = 0
    distances_to_goal = []
    while total_episodes < 100:
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_rewards = []
        total_episodes += 1

        while not done:
            # pass observation to model to get predicted action
            action, _states = model.predict(obs)

            # pass action to env and get info back
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)

            # show the environment on the screen
            env.render()

        distance_to_goal = env.get_distance_to_goal()

        if done and not truncated:
            successful_episodes += 1
            print(f"Episode {total_episodes} successful! Distance to goal: {distance_to_goal}")
        else:
            print(f"Episode {total_episodes} unsuccessful. Distance to goal: {distance_to_goal}")

        distances_to_goal.append(distance_to_goal)

    print(f"Number of successful episodes: {successful_episodes}")
    print(f"Number of total episodes: {total_episodes}")
    print(f"Accuracy = Average distance to goal: {np.mean(distances_to_goal)}")
    print(f"Reliability = Percentage of successful episodes: {successful_episodes / total_episodes * 100}%")

if __name__ == '__main__':
    # train()
    run_model()
