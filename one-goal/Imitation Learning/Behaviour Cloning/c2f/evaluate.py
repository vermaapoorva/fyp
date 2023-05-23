from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import torch

def run_model(name_of_task, scene_name, bottleneck, num_of_runs=10):
    print("Running model")

    env = gym.make("RobotEnv-v2", file_name=scene_name, bottleneck=bottleneck)

    # Load torch model
    image_to_pose_network = ImageToPoseNetworkCoarse(task_name=name_of_task)
    image_to_pose_network.load()
    image_to_pose_network.cuda()
    image_to_pose_network.eval()

    # Evaluate the trained model
    print("Calculating accuracy of model")
    returns = []
    distances_to_goal = []
    orientation_z_diffs = []
    steps_list = []

    for i in trange(num_of_runs):

        obs, _ = env.reset()
        done = False
        total_return = 0
        steps = 0

        while not done:
            # obs = np.expand_dims(obs, axis=0)
            # normalise obs
            obs = obs / 255.0
            z = env.agent.get_position()[2]
            
            image_tensor = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
            z_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(z), 0), 0)

            predicted_action = image_to_pose_network.forward(image_tensor.cuda(), z_tensor.cuda()).detach().cpu().numpy()[0]

            obs, reward, done, truncated, info = env.step(predicted_action)
            total_return += reward
            steps += 1

        print("Episode finished after {} timesteps".format(steps)
            + " with return {}".format(total_return)
            + " and distance to goal {}".format(env.get_distance_to_goal())
            + " and orientation z diff {}".format(env.get_orientation_diff_z()))
        steps_list.append(steps)
        returns.append(total_return)
        distances_to_goal.append(env.get_distance_to_goal())
        orientation_z_diffs.append(env.get_orientation_diff_z())

    print("Accuracy of model (distance): ", np.mean(distances_to_goal))
    print("Accuracy of model (orientation): ", np.mean(orientation_z_diffs))

    env.close()

    return np.mean(returns), np.mean(distances_to_goal), np.mean(orientation_z_diffs)
