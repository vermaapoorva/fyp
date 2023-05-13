import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import robot_env

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

#################################################################################################
###################################   USING THE ENVIRONMENT   ###################################
#################################################################################################

iter = 2
logdir = "logs/logs" + str(iter)
tensorboard_log_dir = "tensorboard_logs"

def train():

    env = make_vec_env("RobotEnv-v0", n_envs=16, vec_env_cls=SubprocVecEnv, monitor_dir=logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 256, 128], qf=[128, 256, 128])
    )

    # model = SAC.load("logs/logs16/best_model.zip", env=env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir)
    model = SAC('CnnPolicy', env, batch_size=4096, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir, device="cuda")

    # Create the callbacks
    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, logdir=logdir)

    # Train the agent
    timesteps = 500000000
    model.learn(total_timesteps=int(timesteps), callback=[save_best_model_callback])
    plot_results([logdir], timesteps, results_plotter.X_TIMESTEPS, "SAC")
    plt.show()

def run_model():

    env = Monitor(gym.make("RobotEnv-v0", headless=True, image_size=64, sleep=0), logdir)

    model_path = f"{logdir}/best_model.zip"
    model = SAC.load(model_path, env=env)

    total_episodes = 0
    successful_episodes = 0
    distances_to_goal = []
    orientation_difference_z = []
    while total_episodes < 100:
        obs, _ = env.reset()
        done = False
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
        orientation_difference_z

        if not truncated:
            successful_episodes += 1
            print(f"Episode {total_episodes} successful! Distance to goal: {distance_to_goal}. Orientation difference z: {orientation_difference_z}")
        
        distances_to_goal.append(distance_to_goal)
        orientation_difference_z.append(orientation_difference_z)

    print(f"Number of successful episodes: {successful_episodes}")
    print(f"Number of total episodes: {total_episodes}")
    print(f"Distance Accuracy = Average distance to goal: {np.mean(distances_to_goal)}")
    print(f"Orientation Accuracy = Average orientation difference z: {np.mean(orientation_difference_z)}")
    print(f"Reliability = Percentage of successful episodes (out of total): {successful_episodes / total_episodes * 100}%")

if __name__ == '__main__':
    train()
    # run_model()
