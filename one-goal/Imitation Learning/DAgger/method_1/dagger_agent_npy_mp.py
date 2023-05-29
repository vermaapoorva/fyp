from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import os
import pickle
import torch
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

def train_model(task_name, scene_file_name, bottleneck, hyperparameters, checkpoint_path=None, start_iteration=0):

    amount_of_training_data = hyperparameters["amount_of_data"]
    num_dagger_iterations = hyperparameters["num_dagger_iterations"]

    # 10% validation data, 90% training data
    val_split = 0.1
    total_amount_of_data = int(amount_of_training_data * 10/9)
    amount_of_data_per_iteration = int(total_amount_of_data // num_dagger_iterations)

    print("Total Amount of training data:", amount_of_training_data)
    print("Total amount of data:", total_amount_of_data)
    print("Amount of data per iteration:", amount_of_data_per_iteration)

    print(f"Hyperparameters: {hyperparameters}")

    amount_of_training_data_per_iteration = int(amount_of_training_data // num_dagger_iterations)
    amount_of_validation_data_per_iteration = amount_of_data_per_iteration - amount_of_training_data_per_iteration

    print("Amount of training data per iteration:", amount_of_training_data_per_iteration)
    print("Amount of validation data per iteration:", amount_of_validation_data_per_iteration)

    ##### Create environment #####
    # env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)
    env = make_vec_env("RobotEnv-v2",
                        n_envs=1,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    for i in trange(num_dagger_iterations):

        if i < start_iteration:
            continue

        ##### Load the trainer #####

        image_to_pose_trainer = ImageToPoseTrainerCoarse(task_name=task_name, scene_name=scene_file_name[:-4] ,hyperparameters=hyperparameters)
        dataset_directory = image_to_pose_trainer.dataset_directory
        raw_dataset_directory = dataset_directory + f"raw_data_{i}/"        
        raw_training_data_directory = raw_dataset_directory + "training_data/"
        raw_validation_data_directory = raw_dataset_directory + "validation_data/"

        if not os.path.exists(raw_dataset_directory):
            os.makedirs(raw_dataset_directory)
        if not os.path.exists(raw_training_data_directory):
            os.makedirs(raw_training_data_directory)
        if not os.path.exists(raw_validation_data_directory):
            os.makedirs(raw_validation_data_directory)

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

        print(f"Collecting training data for iteration {i}...")

        collect_expert_data(env=env,
                            network=image_to_pose_network,
                            amount_of_data=amount_of_training_data_per_iteration,
                            raw_dataset_directory=raw_training_data_directory,
                            iteration=i)

        print(f"Collecting validation data for iteration {i}...")

        collect_expert_data(env=env,
                            network=image_to_pose_network,
                            amount_of_data=amount_of_validation_data_per_iteration,
                            raw_dataset_directory=raw_validation_data_directory,
                            iteration=i)

        ##### Update model with new expert data #####

        image_to_pose_trainer.update_dataloaders(iteration=i,
                                                amount_of_training_data=amount_of_training_data_per_iteration,
                                                amount_of_validation_data=amount_of_validation_data_per_iteration)

        ##### Train the model #####

        image_to_pose_trainer.train()

def collect_expert_data(env, network, amount_of_data, raw_dataset_directory, iteration):

    amount_of_data_collected = 0

    returns = [[] for _ in range(env.num_envs)]
    distances_to_goal = []
    orientation_diffs_z = []

    while amount_of_data_collected < amount_of_data:

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

            expert_actions = expert_policy(env)

            for i in active_envs_indices:
                image = np.array(obss[i])
                action = expert_actions[i]
                endpoint_height = env.env_method("get_agent_position", indices=i)[0][2]

                image = np.transpose(image, (1, 2, 0))
                plt.imsave(f"{raw_dataset_directory}image_{amount_of_data_collected}.png", image)

                action = np.append(action, endpoint_height)
                np.save(f"{raw_dataset_directory}image_{amount_of_data_collected}.npy", action)
                amount_of_data_collected += 1

                if amount_of_data_collected >= amount_of_data:
                    break

            if amount_of_data_collected >= amount_of_data:
                break

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

        print(f"{amount_of_data_collected}/{amount_of_data} --- {(amount_of_data_collected / amount_of_data) * 100}%")

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