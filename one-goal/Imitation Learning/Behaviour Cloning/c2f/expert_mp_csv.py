import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym

from npy_append_array import NpyAppendArray

import os
import robot_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import numpy as np
from math import pi
from fractions import Fraction
from matplotlib import pyplot as plt

import cv2

import pandas as pd

from tqdm import trange


def expert_policy_to_bottleneck(env, bottleneck):
    agent_positions = env.env_method("get_agent_position")
    agent_orientations = env.env_method("get_agent_orientation")

    goal_position = bottleneck[:3]
    goal_orientation = [-np.pi, 0, bottleneck[3]]

    x_diffs = []
    y_diffs = []
    z_diffs = []
    orientation_diffs_z = []
    actions = []

    for i in range(env.num_envs):
        x_diffs.append(goal_position[0] - agent_positions[i][0])
        y_diffs.append(goal_position[1] - agent_positions[i][1])
        z_diffs.append(goal_position[2] - agent_positions[i][2])
        orientation_diffs_z.append(goal_orientation[2] - agent_orientations[i][2])

        if orientation_diffs_z[i] < -np.pi:
            orientation_diffs_z[i] += 2 * np.pi
        elif orientation_diffs_z[i] > np.pi:
            orientation_diffs_z[i] -= 2 * np.pi
        
    for i in range(env.num_envs):
        actions.append(np.array([x_diffs[i], y_diffs[i], z_diffs[i], orientation_diffs_z[i]], dtype=np.float32))

    return actions

def expert_policy_to_target(env):

    agent_positions = env.env_method("get_agent_position")
    agent_orientations = env.env_method("get_agent_orientation")

    goal_positions = env.get_attr("goal_pos")
    goal_orientations = env.get_attr("goal_orientation")

    x_diffs = []
    y_diffs = []
    z_diffs = []
    orientation_diffs_z = []
    actions = []

    for i in range(env.num_envs):
        x_diffs.append(goal_positions[i][0] - agent_positions[i][0])
        y_diffs.append(goal_positions[i][1] - agent_positions[i][1])
        z_diffs.append(goal_positions[i][2] - agent_positions[i][2])
        orientation_diffs_z.append(goal_orientations[i][2] - agent_orientations[i][2])

        if orientation_diffs_z[i] < -np.pi:
            orientation_diffs_z[i] += 2 * np.pi
        elif orientation_diffs_z[i] > np.pi:
            orientation_diffs_z[i] -= 2 * np.pi

        orientation_diffs_z[i] = np.clip(orientation_diffs_z[i], -0.02 * np.pi, 0.02 * np.pi)

    for i in range(env.num_envs):
        actions.append(np.array([x_diffs[i], y_diffs[i], z_diffs[i], orientation_diffs_z[i]], dtype=np.float32))
    
    # print("actions to target: ", actions)

    return actions

def collect_data(scene_file_name, bottleneck, num_of_samples, task_name, start_index=0):

    # If behavioural cloning directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning")
    
    # If hyperparameter directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/c2f"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/c2f")

    # If expert_data directory doesn't exist create it
    logdir = "/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/"

    # If expert_data directory doesn't exist create it
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # scene_file_name = scene_file_name[:-4]
    logdir += task_name + "/"

    # If scene directory doesn't exist create it
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    print("Collecting data for scene:", scene_file_name)
    print("Bottleneck:", bottleneck)
    print("Number of samples:", num_of_samples)

    # env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)
    env = make_vec_env("RobotEnv-v2",
                        n_envs=16,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    # scene_file_name = scene_file_name[:-4]
    # output_file = task_name + ".pkl"
    actions_file = f"{task_name}_{start_index}_actions.npy"
    heights_file = f"{task_name}_{start_index}_heights.npy"
    images_file = f"{task_name}_{start_index}_images.npy"

    returns = [[] for _ in range(env.num_envs)]
    amount_of_data_collected = 0
    distances_to_goal = []
    orientation_diffs_z = []
    # images = []

    translation_noise = 0.025
    rotation_noise = 0.03*np.pi
    data = []
    amount_of_actions_collected = 0
    amount_of_heights_collected = 0

    # Get amount of data in npy file if they exist
    if os.path.exists(logdir + actions_file):
        saved_actions = np.load(logdir + actions_file)
        amount_of_actions_collected = len(saved_actions)
    
    if os.path.exists(logdir + heights_file):
        saved_heights = np.load(logdir + heights_file)
        amount_of_heights_collected = len(saved_heights)

    if amount_of_actions_collected != amount_of_heights_collected:
        print("Amount of actions and heights collected are not equal!")
        print("Amount of actions collected: ", amount_of_actions_collected)
        print("Amount of heights collected: ", amount_of_heights_collected)
        return

    amount_of_data_collected = amount_of_actions_collected
    print("amount of data already collected: ", amount_of_data_collected)

    while amount_of_data_collected < num_of_samples:

        obss = env.reset()
        dones = np.zeros((env.num_envs,), dtype=bool)
        total_return = np.zeros((env.num_envs,))
        steps = 0
        targets = []

        # set goal for each env
        for i in range(env.num_envs):
            target = np.copy(bottleneck)
            target[0] += np.random.uniform(-translation_noise, translation_noise)
            target[1] += np.random.uniform(-translation_noise, translation_noise)
            # target[2] += np.random.uniform(0, translation_noise)
            target[3] += np.random.uniform(-rotation_noise, rotation_noise)

            target[0] = np.clip(target[0], -0.1, 0.1)
            target[1] = np.clip(target[1], -0.1, 0.1)

            goal_pos = target[:3]
            goal_orientation = [-np.pi, 0, target[3]]

            env.env_method("set_goal", indices=i, goal_pos=goal_pos, goal_orientation=goal_orientation)

        images = []
        actions = []
        heights = []

        while not np.all(dones):
            
            active_envs = np.logical_not(dones)
            active_envs_indices = [i for i in range(env.num_envs) if active_envs[i]]
            
            expert_actions_to_bottleneck = expert_policy_to_bottleneck(env, bottleneck)
            expert_actions_to_target = expert_policy_to_target(env)

            for i in active_envs_indices:
                image = obss[i]
                action = expert_actions_to_bottleneck[i]
                endpoint_height = env.env_method("get_agent_position", indices=i)[0][2]

                images.append(image)
                actions.append(action)
                heights.append(endpoint_height)     

            # Get expert actions to target with current observation
            active_expert_actions_to_target = [expert_actions_to_target[i] if active_envs[i] else None for i in range(env.num_envs)]
            env.step_async(active_expert_actions_to_target)
            obss, rewards, active_dones, _ = env.step_wait()

            # Update reward for each env
            for i in range(env.num_envs):
                if active_envs[i]:
                    total_return[i] += rewards[i]
                    steps += 1

            # Update dones
            dones = np.logical_or(dones, active_dones)
                
        # Append distances to goal and orientation z diffs
        for info in env.reset_infos:
            distances_to_goal.append(info["final_distance"])
            orientation_diffs_z.append(info["final_orientation"])

        # Append returns
        for i in range(env.num_envs):
            returns[i].append(total_return[i])

        num_of_images_this_trajectory = 0
        # Save images to img dir with index amount_of_data_collected+start_index
        with NpyAppendArray(logdir + images_file) as image_file_npy:
            for i in range(len(images)):
                image = images[i]
                image = image.astype(np.float32)
                image /= 255.0
                image = np.expand_dims(image, axis=0)
                image_file_npy.append(image)
                # print("image shape: ", image.shape)

                amount_of_data_collected += 1
                num_of_images_this_trajectory += 1
                if amount_of_data_collected >= num_of_samples:
                    break
        
        # Append actions and heights to file
        with NpyAppendArray(logdir + actions_file) as action_file_npy:
            action_file_npy.append(np.array(actions[:num_of_images_this_trajectory]))
        with NpyAppendArray(logdir + heights_file) as heights_file_npy:
            heights_file_npy.append(np.array(heights[:num_of_images_this_trajectory]))

        print(f"{amount_of_data_collected}/{num_of_samples} --- {(amount_of_data_collected/num_of_samples)*100}%")

    print("done collecting data")
    # print mean and std of returns, distances to goal and orientation z diffs
    print("mean return:", np.mean(returns))
    print("mean distance to goal:", np.mean(distances_to_goal))
    print("mean orientation z diff:", np.mean(orientation_diffs_z))

    env.close()

if __name__ == "__main__":
    # scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
    #         ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
    #         ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
    #         ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
    #         ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
    #         ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

    scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
            ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]]]
            
            # ["cutlery_block_scene.ttt", [-0.03, 0.01, 0.768, 0.351]],
            # ["cutlery_block_scene.ttt", [0.025, -0.045, 0.79, -0.424]]]

    # scenes = [["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
    #         ["wooden_block_scene.ttt", [-0.0253, 0.0413, 0.791, -2.164]],
    #         ["wooden_block_scene.ttt", [0.0321, 0.0123, 0.782, -0.262]]]

    # Collect 1M samples for each scene
    num_of_samples = 1000000
    scene_index = 1
    run_index = 9
    scene_name = scenes[scene_index][0].split(".")[0]
    task_name = f"{scene_name}"
    collect_data(scene_file_name=scenes[scene_index][0],
                    bottleneck=scenes[scene_index][1],
                    num_of_samples=num_of_samples,
                    task_name=task_name,
                    start_index=num_of_samples*run_index)
