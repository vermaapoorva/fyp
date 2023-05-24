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

def expert_policy_to_bottleneck(env, bottleneck):

    # print("calculating expert policy to bottleneck: ", bottleneck)

    agent_position = env.agent.get_position()
    agent_orientation = env.agent.get_orientation()
    goal_position = bottleneck[:3]
    goal_orientation = [-np.pi, 0, bottleneck[3]]

    # print("agent_position: ", agent_position)
    # print("agent_orientation: ", agent_orientation)
    # print("goal_position: ", goal_position)
    # print("goal_orientation: ", goal_orientation)

    x_diff = goal_position[0] - agent_position[0]
    y_diff = goal_position[1] - agent_position[1]
    z_diff = goal_position[2] - agent_position[2]
    orientation_diff_z = goal_orientation[2] - agent_orientation[2]

    if orientation_diff_z < -np.pi:
        orientation_diff_z += 2 * np.pi
    elif orientation_diff_z > np.pi:
        orientation_diff_z -= 2 * np.pi

    action = np.array([x_diff, y_diff, z_diff, orientation_diff_z], dtype=np.float32)
    # print("action: ", action)
    return action

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

    orientation_diff_z = np.clip(orientation_diff_z, -0.03 * np.pi, 0.03 * np.pi)

    action = np.array([x_diff, y_diff, z_diff, orientation_diff_z], dtype=np.float32)
    return action


def collect_data(scene_file_name, bottleneck, num_of_samples=20):

    # If behavioural cloning directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning")
    
    # If hyperparameter directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/c2f"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/c2f")

    # If expert_data directory doesn't exist create it
    if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data"):
        os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data")

    print("Collecting data for scene:", scene_file_name)
    print("Bottleneck:", bottleneck)
    print("Number of samples:", num_of_samples)

    logdir = "/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/"
    
    env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)

    scene_file_name = scene_file_name[:-4]
    output_file = str(num_of_samples) + "_expert_data_" + scene_file_name + ".pkl"

    returns = []
    distances_to_goal = []
    orientation_z_diffs = []
    amount_of_data_collected = 0
    images = []

    # Get amount of data in file
    if os.path.exists(logdir + output_file) and os.stat(logdir + output_file).st_size != 0:
        with open(logdir + output_file, "rb") as f:
            amount_of_data_collected = len(pickle.load(f))
            print("amount of data:", amount_of_data_collected)

    translation_noise = 0.05
    rotation_noise = 0.03*np.pi

    while amount_of_data_collected < num_of_samples:
        data = []

        obs, _ = env.reset()
        done = False
        total_return = 0
        steps = 0

        target = np.copy(bottleneck)
        target[0] += np.random.uniform(-translation_noise, translation_noise)
        target[1] += np.random.uniform(-translation_noise, translation_noise)
        # target[2] += 
        target[3] += np.random.uniform(-rotation_noise, rotation_noise)

        env.goal_pos = target[:3]
        env.goal_orientation = [-np.pi, 0, target[3]]
        
        env.set_goal(env.goal_pos, env.goal_orientation)

        while not done:
            expert_action_to_bottleneck = expert_policy_to_bottleneck(env, bottleneck)

            images.append(obs)

            # change obs type to float32
            obs = obs.astype(np.float32)
            obs = obs / 255.0

            data.append({"image": obs, "action": expert_action_to_bottleneck, "endpoint_height": env.agent.get_position()[2]})

            expert_action_to_target = expert_policy(env)

            obs, reward, done, truncated, info = env.step(expert_action_to_target)
            total_return += reward
            steps += 1

        print("Episode completed in", steps, "steps.")

        # If file doesn't exist or is empty, create new one
        if not os.path.exists(logdir + output_file) or os.stat(logdir + output_file).st_size == 0:
            with open(logdir + output_file, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        # Else append to existing file
        else:
            with open(logdir + output_file, "rb") as f:
                existing_data = pickle.load(f)
            existing_data.extend(data)
            with open(logdir + output_file, "wb") as f:
                pickle.dump(existing_data, f, pickle.HIGHEST_PROTOCOL)

        returns.append(total_return)
        distances_to_goal.append(env.get_distance_to_goal())
        orientation_z_diffs.append(env.get_orientation_diff_z())
        
        # Get amount of data in file
        with open(logdir + output_file, "rb") as f:
            amount_of_data_collected = len(pickle.load(f))
            print("amount of data:", amount_of_data_collected)

    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))
    print("mean distance to goal", np.mean(distances_to_goal))
    print("std of distance to goal", np.std(distances_to_goal))
    print("mean orientation z diff", np.mean(orientation_z_diffs))
    print("std of orientation z diff", np.std(orientation_z_diffs))

    # # Save data to file
    # with open(logdir + str(len(observations)) + "_" + output_file, "wb") as f:
    #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # # Create images dir if not exist
    if not os.path.exists(os.path.join(logdir, "images")):
        os.makedirs(os.path.join(logdir, "images"))
    # Save observations as images
    for i in range(len(images)):
        image = images[i]
        # make it channel last
        image = np.transpose(image, (1, 2, 0))
        # save image
        image_file_name = "images/image_" + str(i) + ".png"
        image_file_path = os.path.join(logdir, image_file_name)
        plt.imsave(image_file_path, image)
    env.close()

if __name__ == "__main__":
    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
            ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
            ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
            ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
            ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
            ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

    # Collect 1M samples for each scene
    num_of_samples = 12000
    for scene in scenes[0:1]:
        collect_data(scene[0], scene[1], num_of_samples)