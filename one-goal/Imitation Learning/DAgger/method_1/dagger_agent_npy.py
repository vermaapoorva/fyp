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
    env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)

    for i in trange(num_dagger_iterations):

        if i < start_iteration:
            continue

        ##### Load the trainer #####

        image_to_pose_trainer = ImageToPoseTrainerCoarse(task_name=task_name, scene_name=scene_file_name[:-4] ,hyperparameters=hyperparameters)
        action_file = image_to_pose_trainer.image_to_pose_dataset.action_file
        height_file = image_to_pose_trainer.image_to_pose_dataset.height_file
        image_file = image_to_pose_trainer.image_to_pose_dataset.image_file

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
                            image_file=image_file)

        ##### Update model with new expert data #####

        image_to_pose_trainer.update_dataloaders(amount_of_data_per_iteration * (i+1))

        ##### Train the model #####

        image_to_pose_trainer.train()

def collect_expert_data(env, network, amount_of_data_per_iteration, action_file, height_file, image_file):

    amount_of_data_collected = 0
    steps_list = []

    while amount_of_data_collected < amount_of_data_per_iteration:
        done = False
        obs, _ = env.reset()
        total_return = 0
        steps = 0

        images = []
        actions = []
        heights = []

        while not done:

            image = obs
            expert_action = expert_policy(env)
            z = env.agent.get_position()[2]

            images.append(image)
            actions.append(expert_action)
            heights.append(z)
            
            amount_of_data_collected += 1

            obs = obs.astype(np.float32)
            obs /= 255.0

            image_tensor = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
            z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)

            predicted_action = network.forward(image_tensor.cuda(), z_tensor.cuda()).detach().cpu().numpy()[0]

            obs, reward, done, truncated, info = env.step(predicted_action)
            total_return += reward
            steps += 1
            
            if amount_of_data_collected >= amount_of_data_per_iteration:
                break

        with NpyAppendArray(image_file) as image_file_npy:
            for image in images:
                image = image.astype(np.float32)
                image /= 255.0
                image = np.expand_dims(image, axis=0)
                image_file_npy.append(image)

        with NpyAppendArray(action_file) as action_file_npy:
            action_file_npy.append(np.array(actions))
        
        with NpyAppendArray(height_file) as height_file_npy:
            height_file_npy.append(np.array(heights))

        print(f"Collecting data: {amount_of_data_collected}/{amount_of_data_per_iteration}")

        steps_list.append(steps)

    print(f"Collected {amount_of_data_collected} new data points.")
    print("Steps:", steps_list)
    return amount_of_data_collected

def expert_policy(env):
    agent_position = env.agent.get_position()
    agent_orientation = env.agent.get_orientation()
    goal_position = env.goal_pos
    goal_orientation = env.goal_orientation

    x_diff = goal_position[0] - agent_position[0]
    y_diff = goal_position[1] - agent_position[1]
    z_diff = goal_position[2] - agent_position[2]
    orientation_diff_z = goal_orientation[2] - agent_orientation[2]

    if orientation_diff_z < -np.pi:
        orientation_diff_z += 2 * np.pi
    elif orientation_diff_z > np.pi:
        orientation_diff_z -= 2 * np.pi

    action = np.array([x_diff, y_diff, z_diff, orientation_diff_z], dtype=np.float32)
    return action