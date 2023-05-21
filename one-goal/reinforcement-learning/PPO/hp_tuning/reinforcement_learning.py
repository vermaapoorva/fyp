import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import robot_env
from zipfile import ZipFile
import pickle

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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import TensorBoardOutputFormat

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def get_number_of_parameters(net_arch, env):
    # Load the model from the zip file
    # model = PPO.load(model_file_path)

    # Create CNN PPO model with net_arch policy+vf:

    model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)))

    num_of_params = 0
    policy_dict = model.get_parameters().get("policy")
    for key, value in policy_dict.items():
        
        # Add numel if key contains pi_features_extractor, mlp_extractor.policy_net, action_net
        if "pi_features_extractor" in key or "mlp_extractor.policy_net" in key or "action_net" in key:
            num_of_params += value.numel()
        
    print("Number of parameters: ", num_of_params)
    return num_of_params

def train(scene_file_name, bottleneck, hyperparameters, hyperparam_i, scene_num):

    print("Training on scene: " + scene_file_name)
    print("Bottleneck x: " + str(bottleneck[0]))
    print("Bottleneck y: " + str(bottleneck[1]))
    print("Bottleneck z: " + str(bottleneck[2]))
    print("Bottleneck z angle: " + str(bottleneck[3]))
    print("Hyperparameters: " + str(hyperparameters))

    logdir = f"/vol/bitbucket/av1019/PPO/logs/hp_{hyperparam_i}_scene_{scene_num}/"
    tensorboard_log_dir = f"/vol/bitbucket/av1019/PPO/tensorboard_logs/hp_{hyperparam_i}_scene_{scene_num}/"

    env = make_vec_env("RobotEnv-v2",
                        n_envs=16,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))
    
    eval_env = make_vec_env("RobotEnv-v2",
                        n_envs=1,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    net_arch = hyperparameters["net_arch"]
    batch_size = hyperparameters["batch_size"]
    n_steps = hyperparameters["n_steps"]

    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=net_arch, vf=net_arch)
    )

    model = PPO('CnnPolicy', env, batch_size=batch_size, n_steps=n_steps, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir)

    # Create the callbacks
    eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=logdir,
                                    log_path=logdir,
                                    eval_freq=20000,
                                    # callback_on_new_best=callback_on_best,
                                    deterministic=True,
                                    verbose=1)

    # Train the agent for 7.5M timesteps
    timesteps = 7500000
    model.learn(total_timesteps=int(timesteps), callback=[eval_callback])
    model.save(f"{logdir}/final_model.zip")

def run_model():

    env = Monitor(gym.make("RobotEnv-v2", headless=True, image_size=64, sleep=0), logdir)

    model_path = f"{logdir}/best_model.zip"
    model = PPO.load(model_path, env=env)

    total_episodes = 0
    successful_episodes = 0
    distances_to_goal = []
    orientation_differences_z = []
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
        orientation_difference_z = env.get_orientation_diff_z()

        if not truncated:
            successful_episodes += 1
            print(f"Episode {total_episodes} successful! Distance to goal: {distance_to_goal}. Orientation difference z: {orientation_difference_z}")
        
        distances_to_goal.append(distance_to_goal)
        orientation_differences_z.append(orientation_difference_z)

    print(f"Number of successful episodes: {successful_episodes}")
    print(f"Number of total episodes: {total_episodes}")
    print(f"Distance Accuracy = Average distance to goal: {np.mean(distances_to_goal)}")
    print(f"Orientation Accuracy = Average orientation difference z: {np.mean(orientation_differences_z)}")
    print(f"Reliability = Percentage of successful episodes (out of total): {successful_episodes / total_episodes * 100}%")