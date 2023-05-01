import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2

import custom_impl_7

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

iter = 10
logdir = "logs" + str(iter)
tensorboard_log_dir = "tensorboard_logs"
tensorboard_callback = TensorBoardOutputFormat(tensorboard_log_dir + "/Average final reward_" + str(iter))

def train():

    env = make_vec_env("RobotEnv7-v0", n_envs=16, vec_env_cls=SubprocVecEnv, monitor_dir=logdir)
    # env = gym.make("RobotEnv7-v0", scene_file=SCENE_FILE)
    # env = Monitor(env, logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    # policy_kwargs = dict(net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]))
    
    # parameters: {'gamma': 0.0008345698046059239,
    #              'max_grad_norm': 0.32642320598728214,
    #              'gae_lambda': 0.0054705510876348375,
    #              'exponent_n_steps': 12, 'lr': 0.0009719718719738601,
    #              'exponent_batch_size': 9,
    #              'ent_coef': 1.4257709536587286e-06,
    #              'ortho_init': True,
    #              'net_arch': 'small',
    #              'activation_fn': 'tanh'}

    gamma = 1.0 - 0.0008345698046059239
    max_grad_norm = 0.32642320598728214
    gae_lambda = 1.0 - 0.0054705510876348375
    n_steps = 2 ** 12
    learning_rate = 0.0009719718719738601
    batch_size = 2 ** 9
    ent_coef = 1.4257709536587286e-06
    ortho_init = True
    net_arch = "small"
    activation_fn = "tanh"
    
    if net_arch == "small":
        net_arch = [{"pi": [128, 128, 128], "vf": [128, 128, 128]}]
    elif net_arch == "medium":
        net_arch = [{"pi": [128, 256, 256, 128], "vf": [128, 256, 256, 128]}]
    elif net_arch == "large":
        net_arch = dict(pi=[128, 256, 512, 256, 128], vf=[128, 256, 512, 256, 128])
    else:
        net_arch = dict(pi=[128, 256, 512, 1024, 1024, 512, 256, 128], vf=[128, 256, 512, 1024, 1024, 512, 256, 128])

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    kwargs = {
       "n_steps": n_steps,
       "gamma": gamma,
       "gae_lambda": gae_lambda,
       "learning_rate": learning_rate,
       "batch_size": batch_size,
       "ent_coef": ent_coef,
       "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }


    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, **kwargs)

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

        if info.get("success"):
            successful_episodes += 1
            print(f"Episode {total_episodes} successful! Distance to goal: {distance_to_goal}")
        
        distances_to_goal.append(distance_to_goal)

    print(f"Number of successful episodes: {successful_episodes}")
    print(f"Number of total episodes: {total_episodes}")
    print(f"Accuracy = Average distance to goal for valid episodes: {np.mean(distances_to_goal)}")
    print(f"Reliability = Percentage of successful episodes (out of total): {successful_episodes / total_episodes * 100}%")

if __name__ == '__main__':
    train()
    # run_model()
