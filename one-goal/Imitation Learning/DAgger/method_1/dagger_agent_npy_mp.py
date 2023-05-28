from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import os
import pickle
import torch
from npy_append_array import NpyAppendArray

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def train_model(task_name, scene_file_name, bottleneck, hyperparameters, checkpoint_path=None, start_iteration=0):

    amount_of_training_data = hyperparameters["amount_of_data"]
    num_dagger_iterations = hyperparameters["num_dagger_iterations"]

    # 20% validation data, 80% training data
    total_amount_of_data = int(amount_of_training_data * 1.25)
    amount_of_data_per_iteration = int(total_amount_of_data // num_dagger_iterations)

    print("Amount of training data:", amount_of_training_data)
    print("Total amount of data:", total_amount_of_data)
    print("Amount of data per iteration:", amount_of_data_per_iteration)

    print(f"Hyperparameters: {hyperparameters}")

    ##### Create environment #####
    # env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)
    env = make_vec_env("RobotEnv-v2",
                        n_envs=16,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    for i in trange(num_dagger_iterations):

        if i < start_iteration:
            continue

        ##### Load the trainer #####

        image_to_pose_trainer = ImageToPoseTrainerCoarse(task_name=task_name, scene_name=scene_file_name[:-4] ,hyperparameters=hyperparameters)
        action_file = image_to_pose_trainer.image_to_pose_dataset.action_file
        height_file = image_to_pose_trainer.image_to_pose_dataset.height_file
        image_file = get_attr(image_to_pose_trainer.image_to_pose_dataset, f"image{i}_file")
        print("Image file:", image_file)

        print("Action file:", action_file)
        print("Height file:", height_file)
        print("Image file:", image_file)

        ##### Load the model #####

        image_to_pose_network = image_to_pose_trainer.get_network()
        if checkpoint_path is not None:
            image_to_pose_network.load_from_checkpoint(checkpoint_path)
            checkpoint_path = None
        else:
            image_to_pose_network.load()

        image_to_pose_network.cuda()
        image_to_pose_network.eval()

        ##### Collect data from the expert #####

        collect_expert_data(env=env,
                            network=image_to_pose_network,
                            amount_of_data_per_iteration=amount_of_data_per_iteration,
                            action_file=action_file,
                            height_file=height_file,
                            image_file=image_file,
                            iteration=i)

        ##### Update model with new expert data #####

        image_to_pose_trainer.update_dataloaders(amount_of_data_per_iteration * (i+1), iteration=i)

        ##### Train the model #####

        image_to_pose_trainer.train()

def collect_expert_data(env, network, amount_of_data_per_iteration, action_file, height_file, image_file, iteration):

    amount_of_data_collected = 0

    returns = [[] for _ in range(env.num_envs)]
    distances_to_goal = []
    orientation_diffs_z = []

    while amount_of_data_collected < amount_of_data_per_iteration:

        obss = env.reset()
        dones = np.zeros((env.num_envs,), dtype=bool)
        total_return = np.zeros((env.num_envs,))
        steps = 0

        images = []
        actions = []
        heights = []

        while not np.all(dones):

            active_envs = np.logical_not(dones)
            active_envs_indices = [i for i in range(env.num_envs) if active_envs[i]]

            # print("Active envs:", active_envs_indices)

            expert_actions = expert_policy(env)

            for i in active_envs_indices:
                image = obss[i]
                endpoint_height = env.env_method("get_agent_position", indices=i)[0][2]

                images.append(image)
                actions.append(expert_actions[i])
                heights.append(endpoint_height)

            predicted_actions = []

            for i in range(env.num_envs):
                if i not in active_envs_indices:
                    predicted_actions.append(None)
                else:
                    image = obss[i]
                    z = env.env_method("get_agent_position", indices=i)[0][2]
                    image = image.astype(np.float32)
                    image /= 255.0

                    image_tensor = torch.unsqueeze(torch.tensor(image, dtype=torch.float32), 0)
                    z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)

                    predicted_action = network.forward(image_tensor.cuda(), z_tensor.cuda()).detach().cpu().numpy()[0]
                    predicted_actions.append(predicted_action)

            env.step_async(predicted_actions)
            obss, rewards, active_dones, _ = env.step_wait()

            for i in active_envs_indices:
                total_return[i] += rewards[i]
                steps += 1
            
            dones = np.logical_or(dones, active_dones)

        for info in env.reset_infos:
            distances_to_goal.append(info["final_distance"])
            orientation_diffs_z.append(info["final_orientation"])

        for i in range(env.num_envs):
            returns[i].append(total_return[i])

        num_of_images_this_trajectory = 0
        with NpyAppendArray(image_file) as image_array_npy:
            for image in images:
                image = image.astype(np.float32)
                image /= 255.0
                image = np.expand_dims(image, axis=0)
                image_array_npy.append(image)

                amount_of_data_collected += 1
                num_of_images_this_trajectory += 1
                if amount_of_data_collected >= amount_of_data_per_iteration:
                    break
        
        with NpyAppendArray(action_file) as action_file_npy:
            action_file_npy.append(np.array(actions[:num_of_images_this_trajectory]))
        with NpyAppendArray(height_file) as height_file_npy:
            height_file_npy.append(np.array(heights[:num_of_images_this_trajectory]))

        print(f"{amount_of_data_collected}/{amount_of_data_per_iteration} --- {(amount_of_data_collected / amount_of_data_per_iteration) * 100}%")

    print(f"Collected {amount_of_data_collected} new data points.")
    print("Mean return:", np.mean(returns))
    print("Mean distance to goal:", np.mean(distances_to_goal))
    print("Mean orientation diff:", np.mean(orientation_diffs_z))
    return amount_of_data_collected

def expert_policy(env):

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

    for i in range(env.num_envs):
        actions.append(np.array([x_diffs[i], y_diffs[i], z_diffs[i], orientation_diffs_z[i]], dtype=np.float32))

    return actions