from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import os
import pickle
import torch

def train_model(task_name, scene_file_name, bottleneck, hyperparameters, checkpoint_path=None):

    ##### Create environment #####
    env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)

    for i in trange(hyperparameters['num_dagger_iterations']):

        ##### Load the trainer #####

        image_to_pose_trainer = ImageToPoseTrainerCoarse(task_name=task_name, hyperparameters=hyperparameters)
        data_file = image_to_pose_trainer.get_data_file()

        ##### Load the model #####

        image_to_pose_network = image_to_pose_trainer.get_network()
        if checkpoint_path is not None:
            image_to_pose_network.load_from_checkpoint(checkpoint_path)
        else:
            image_to_pose_network.load()

        image_to_pose_network.cuda()
        image_to_pose_network.eval()

        ##### Collect data from the expert #####

        collect_expert_data(env=env,
                            network=image_to_pose_network,
                            num_rollouts=hyperparameters['num_rollouts'],
                            data_file=data_file)

        ##### Update model with new expert data #####

        image_to_pose_trainer.update_dataloaders()

        ##### Train the model #####

        image_to_pose_trainer.train()

def collect_expert_data(env, network, num_rollouts, data_file):

    returns = []
    distances_to_goal = []
    orientation_z_diffs = []
    new_data = []
    steps_list = []

    for i in trange(num_rollouts):
        done = False
        obs, _ = env.reset()
        total_return = 0
        steps = 0
        while not done:
            expert_action = expert_policy(env)

            obs = obs.astype(np.float32)
            obs = obs / 255.0
            z = env.agent.get_position()[2]

            if i==0 and steps==0:
                print("Obs dtype:", obs.dtype)

            new_data.append({"image": obs, "action": expert_action, "endpoint_height": z})

            image_tensor = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
            z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)

            predicted_action = network.forward(image_tensor.cuda(), z_tensor.cuda()).detach().cpu().numpy()[0]

            obs, reward, done, truncated, info = env.step(predicted_action)
            total_return += reward
            steps += 1

        steps_list.append(steps)
        returns.append(total_return)
        distances_to_goal.append(env.get_distance_to_goal())
        orientation_z_diffs.append(env.get_orientation_diff_z())

    # If file does not exist or is empty, create new file/overwrite existing file
    if not os.path.exists(data_file) or os.stat(data_file).st_size == 0:
        with open(data_file, 'wb') as f:
            pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)

    # Otherwise, append to existing file
    else:
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)

        existing_data.extend(new_data)

        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f, pickle.HIGHEST_PROTOCOL)

    print(f"{num_rollouts} rollouts completed.")
    print("Average steps:", np.mean(steps_list))
    print("Average return:", np.mean(returns))
    print("Average distance to goal:", np.mean(distances_to_goal))
    print("Average orientation z diff:", np.mean(orientation_z_diffs))
    print("Steps:", steps_list)

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