import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym

import os
import robot_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import numpy as np
from math import pi
from fractions import Fraction
from matplotlib import pyplot as plt

from tqdm import trange

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


def collect_data(scene_file_name, bottleneck, num_of_samples=20):

    # If behavioural cloning directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning")
    
    # If hyperparameter directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters")

    # If expert_data directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/expert_data"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/expert_data")

    print("Collecting data for scene:", scene_file_name)
    print("Bottleneck:", bottleneck)
    print("Number of samples:", num_of_samples)

    logdir = "/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/expert_data/"
    
    env = Monitor(gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck), "logs/")

    returns = []
    observations = []
    actions = []
    distances_to_goal = []
    orientation_z_diffs = []
    images = []

    while len(observations) < num_of_samples:

        obs, _ = env.reset()
        done = False
        total_return = 0
        steps = 0

        while not done:
            expert_action = expert_policy(env)
            images.append(obs)
            # change obs type to float32
            obs = obs.astype(np.float32)
            obs = obs / 255.0
            if len(observations)==0:
                print(obs)
            observations.append(obs)
            actions.append(expert_action)
            obs, reward, done, truncated, info = env.step(expert_action)
            total_return += reward
            steps += 1

            if(len(observations) % 10000 == 0):
                print("amount of data:", len(observations))
                print(len(observations) / num_of_samples * 100, "%")

        returns.append(total_return)
        distances_to_goal.append(env.get_distance_to_goal())
        orientation_z_diffs.append(env.get_orientation_diff_z())

    # print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))
    # print("distances to goal", distances_to_goal)
    print("mean distance to goal", np.mean(distances_to_goal))
    print("std of distance to goal", np.std(distances_to_goal))
    # print("orientation z diffs", orientation_z_diffs)
    print("mean orientation z diff", np.mean(orientation_z_diffs))
    print("std of orientation z diff", np.std(orientation_z_diffs))

    expert_data = {"observations": np.array(observations),
                     "actions": np.array(actions)}


    # file name without .ttt
    scene_file_name = scene_file_name[:-4]
    output_file = "expert_data_" + scene_file_name + "_" + str(num_of_samples) + ".pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    # # Create images dir if not exist
    # if not os.path.exists(os.path.join(logdir, "images")):
    #     os.makedirs(os.path.join(logdir, "images"))
    # # Save observations as images
    # for i in range(len(images)):
    #     image = images[i]
    #     # make it channel last
    #     image = np.transpose(image, (1, 2, 0))
    #     # save image
    #     image_file_name = "images/image_" + str(i) + ".png"
    #     image_file_path = os.path.join(logdir, image_file_name)
    #     plt.imsave(image_file_path, image)
    env.close()

if __name__ == "__main__":
    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
            ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
            ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
            ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
            ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
            ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

    # Collect 1M samples for each scene
    num_of_samples = 10000
    for scene in scenes[4:]:
        collect_data(scene[0], scene[1], num_of_samples)