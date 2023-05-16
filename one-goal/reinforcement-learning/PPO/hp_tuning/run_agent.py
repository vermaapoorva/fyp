import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
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
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import TensorBoardOutputFormat

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
                        monitor_dir=logdir,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))
    
    eval_env = make_vec_env("RobotEnv-v2",
                        n_envs=1,
                        vec_env_cls=SubprocVecEnv,
                        monitor_dir=logdir,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    net_arch = hyperparameters["net_arch"]
    batch_size = hyperparameters["batch_size"]
    n_steps = hyperparameters["n_steps"]

    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch)
    )

    # Continue training if a model was already trained
    model_path = f"{logdir}/best_model.zip"
    if os.path.exists(model_path):
        reset_num_timesteps = False
        model = PPO.load(model_path, env=env, tensorboard_log=tensorboard_log_dir)
    else:
        reset_num_timesteps = True
        model = PPO('CnnPolicy', env, batch_size=batch_size, n_steps=n_steps, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log_dir)

    # Create the callbacks
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=logdir,
                                    log_path=logdir,
                                    eval_freq=1000,
                                    # callback_on_new_best=callback_on_best,
                                    verbose=1)

    # Train the agent for 10M timesteps or until it reaches the reward threshold
    timesteps = 10000000
    model.learn(total_timesteps=int(timesteps), callback=[eval_callback], reset_num_timesteps=reset_num_timesteps)

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

if __name__ == '__main__':

    hyperparameters = [{"net_arch": [128, 128], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [128, 128], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [128, 128], "batch_size": 8192, "n_steps": 4096}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [128, 128], "batch_size": 16384, "n_steps": 8192}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [128, 256, 128], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [128, 256, 128], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [128, 256, 128], "batch_size": 8192, "n_steps": 4096}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [128, 256, 128], "batch_size": 16384, "n_steps": 8192}, # rollout buffer size: 8192*16*2 = 262144, updates: 262144/16384 = 16
                    ]

    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
                ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
                ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
                ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
                ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
                ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

    # If PPO directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO"):
        os.makedirs("/vol/bitbucket/av1019/PPO")

    # If hyperparameters directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO/hyperparameters"):
        os.makedirs("/vol/bitbucket/av1019/PPO/hyperparameters")

    hp_indexes = [0, 1]
    # hp_indexes = [2, 3]


    for i in hp_indexes:
        scene_num = 1
        for scene_file_name, bottleneck in scenes:

            # Save the hyperparameters, scene name and bottleneck to a file, create it if it doesn't exist
            with open(f"/vol/bitbucket/av1019/PPO/hyperparameters/hps_{i}_scene_{scene_num}.txt", "w+") as f:
                f.write(f"Scene: {scene_file_name}\n")
                f.write(f"Bottleneck: {bottleneck}\n")
                f.write(f"Hyperparameters: {hyperparameters[i]}\n")

            # Train the model
            print(f"Training model {i} on scene {scene_num} with hyperparameters {hyperparameters[i]}")
            train(scene_file_name, bottleneck, hyperparameters[i], i, scene_num)
            
            scene_num += 1

    # run_model()
