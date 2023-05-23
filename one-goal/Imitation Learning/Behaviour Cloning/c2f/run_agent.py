from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import torch

or_coeffs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]

name_of_task = f"orientation_coeff_"

trainer = ImageToPoseTrainerCoarse(task_name=name_of_task)
trainer.train()


def run_model(env, name_of_task, scene_name, bottleneck, num_of_runs=10):
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


scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
            ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
            ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
            ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
            ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
            ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

scene_index = 0

print(run_model(env=env, num_of_runs=5))
