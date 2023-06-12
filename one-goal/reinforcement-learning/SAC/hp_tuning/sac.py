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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def get_number_of_parameters(net_arch, env):
    # Load the model from the zip file
    # model = SAC.load(model_file_path)

    # Create CNN SAC model with net_arch policy+vf:

    model = SAC('CnnPolicy', env, verbose=1, policy_kwargs=dict(net_arch=dict(pi=net_arch, qf=net_arch)), buffer_size=100000)

    num_of_params = 0

    # print("model.get_parameters(): ", model.get_parameters())

    policy_dict = model.get_parameters().get("policy")
    for key, value in policy_dict.items():
        
        # Add numel if key contains pi_features_extractor, mlp_extractor.policy_net, action_net
        if "critic" not in key:
            num_of_params += value.numel()
        
    print("Number of parameters: ", num_of_params)
    return num_of_params

def train(scene_file_name, bottleneck, seed, hyperparameters, task_name):

    print("Training on scene: " + scene_file_name)
    print("Bottleneck x: " + str(bottleneck[0]))
    print("Bottleneck y: " + str(bottleneck[1]))
    print("Bottleneck z: " + str(bottleneck[2]))
    print("Bottleneck z angle: " + str(bottleneck[3]))
    print("Hyperparameters: " + str(hyperparameters))

    logdir = f"/vol/bitbucket/av1019/SAC/logs/{task_name}/"
    tensorboard_log_dir = f"/vol/bitbucket/av1019/SAC/tensorboard_logs/{task_name}/"

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
    buffer_size = hyperparameters["buffer_size"]

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, qf=net_arch)
    )

    model = SAC('CnnPolicy', env, seed=seed, buffer_size=buffer_size, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir)
    # model = SAC.load(f"{logdir}/best_model.zip", env=env, tensorboard_log=tensorboard_log_dir)

    scene_name = scene_file_name.split(".")[0]

    # Create the callbacks
    eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=logdir,
                                    log_path=logdir,
                                    eval_freq=20000,
                                    deterministic=True,
                                    verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=31250, save_path=logdir,
                                         name_prefix=f'final_model_{scene_name}')

    # Train the agent for 1.5M timesteps
    timesteps = 10000000
    model.learn(total_timesteps=int(timesteps), callback=[eval_callback, checkpoint_callback])
    # model.learn(total_timesteps=int(timesteps), callback=[eval_callback, checkpoint_callback], reset_num_timesteps=False)
    model.save(f"{logdir}/final_model.zip")

    env.close()
    eval_env.close()

def run_model(task_name, scene_file_name, bottleneck, num_of_runs=30, amount_of_data=None):

    logdir = f"/vol/bitbucket/av1019/SAC/logs/{task_name}/"

    env = Monitor(gym.make("RobotEnv-v2", headless=True, image_size=64, sleep=0, file_name=scene_file_name, bottleneck=bottleneck), logdir)

    scene_name = scene_file_name.split(".")[0]
    if amount_of_data is not None:
        model_path = f"{logdir}final_model_{scene_name}_{amount_of_data}_steps.zip"
    else:
        model_path = f"{logdir}best_model.zip"

    print("model path: ", model_path)
    model = SAC.load(model_path, env=env)

    total_episodes = 0
    successful_episodes = 0
    distances_to_goal = []
    orientation_differences_z = []
    while total_episodes < num_of_runs:
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
        else:
            print(f"Episode {total_episodes} unsuccessful! Distance to goal: {distance_to_goal}. Orientation difference z: {orientation_difference_z}")

        distances_to_goal.append(distance_to_goal)
        orientation_differences_z.append(orientation_difference_z)

    print(f"Number of successful episodes: {successful_episodes}")
    print(f"Number of total episodes: {total_episodes}")
    print(f"Distance Accuracy = Average distance to goal: {np.mean(distances_to_goal)}")
    print(f"Orientation Accuracy = Average orientation difference z: {np.mean(orientation_differences_z)}")
    print(f"Reliability = Percentage of successful episodes (out of total): {successful_episodes / total_episodes * 100}%")

    env.close()

    return distances_to_goal, orientation_differences_z, successful_episodes, successful_episodes / total_episodes * 100