from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import os
import pickle
import torch

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

        image_to_pose_trainer = ImageToPoseTrainerCoarse(task_name=task_name, hyperparameters=hyperparameters)
        data_file = image_to_pose_trainer.get_data_file()

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

        total_amount_of_data = collect_expert_data(env=env,
                                                    network=image_to_pose_network,
                                                    amount_of_data_per_iteration=amount_of_data_per_iteration,
                                                    data_file=data_file)

        ##### Update model with new expert data #####

        image_to_pose_trainer.update_dataloaders(amount_of_data_per_iteration * (i+1))

        ##### Train the model #####

        image_to_pose_trainer.train()

def collect_expert_data(env, network, amount_of_data_per_iteration, data_file, whole_trajectory=False):

    new_data = []
    steps_list = []

    while len(new_data) < amount_of_data_per_iteration:
        done = False
        obs, _ = env.reset()
        total_return = 0
        steps = 0
        while not done:
            if whole_trajectory:
                print("Collecting data for whole trajectory")

                # Save the current state of the environment
                agent_position = env.agent.get_position()
                agent_orientation = env.agent.get_orientation()
                original_obs = obs.copy()

                # Run the expert policy until the end of the episode
                while not done:
                    expert_action = expert_policy(env)

                    obs = obs.astype(np.float32)
                    obs = obs / 255.0
                    z = env.agent.get_position()[2]

                    new_data.append({"image": obs, "action": expert_action, "endpoint_height": z})

                    # Clip orientation diff z to +- 0.03*np.pi
                    clipped_expert_action = expert_action.copy()
                    clipped_expert_action[3] = np.clip(clipped_expert_action[3], -0.03*np.pi, 0.03*np.pi)

                    obs, reward, done, truncated, info = env.step(clipped_expert_action)
                    steps += 1

                    if len(new_data) >= amount_of_data_per_iteration:
                        break

                # Restore the environment to its original state
                env.set_agent(agent_position, agent_orientation)
                obs = original_obs

                obs = obs.astype(np.float32)
                obs = obs / 255.0
                z = env.agent.get_position()[2]

            else:
                print("Collecting data for one step")

                expert_action = expert_policy(env)
                steps += 1

                obs = obs.astype(np.float32)
                obs = obs / 255.0
                z = env.agent.get_position()[2]

                new_data.append({"image": obs, "action": expert_action, "endpoint_height": z})

            image_tensor = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
            z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)

            predicted_action = network.forward(image_tensor.cuda(), z_tensor.cuda()).detach().cpu().numpy()[0]

            obs, reward, done, truncated, info = env.step(predicted_action)
            total_return += reward
            
            if len(new_data) >= amount_of_data_per_iteration:
                break

        steps_list.append(steps)

    total_amount_of_data = 0

    # If file does not exist or is empty, create new file/overwrite existing file
    if not os.path.exists(data_file) or os.stat(data_file).st_size == 0:
        with open(data_file, 'wb') as f:
            pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
        total_amount_of_data = len(new_data)

    # Otherwise, append to existing file
    else:
        with open(data_file, 'rb') as f:
            existing_data = pickle.load(f)

        existing_data.extend(new_data)

        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f, pickle.HIGHEST_PROTOCOL)
        
        total_amount_of_data = len(existing_data)

    print(f"Collected {len(new_data)} new data points.")
    print("Steps:", steps_list)
    print(f"Total amount of data: {total_amount_of_data}")
    return total_amount_of_data

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