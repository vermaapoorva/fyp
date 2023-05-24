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

    goal_positions = env.env_attr("goal_pos")
    goal_orientations = env.env_attr("goal_orientation")

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

        orientation_diff_z[i] = np.clip(orientation_diff_z[i], -0.03 * np.pi, 0.03 * np.pi)

    for i in range(env.num_envs):
        actions.append(np.array([x_diffs[i], y_diffs[i], z_diffs[i], orientation_diffs_z[i]], dtype=np.float32))
    
    return actions

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
    
    # env = gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck)
    env = make_vec_env("RobotEnv-v2",
                        n_envs=16,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    scene_file_name = scene_file_name[:-4]
    output_file = str(num_of_samples) + "_expert_data_" + scene_file_name + "2.pkl"

    returns = [[] for _ in range(env.num_envs)]
    amount_of_data_collected = 0
    distances_to_goal = []
    orientation_diffs_z = []

    # Get amount of data in file
    if os.path.exists(logdir + output_file) and os.stat(logdir + output_file).st_size != 0:
        with open(logdir + output_file, "rb") as f:
            amount_of_data_collected = len(pickle.load(f))
            print("amount of data:", amount_of_data_collected)

    translation_noise = 0.05
    rotation_noise = 0.03*np.pi

    while amount_of_data_collected < num_of_samples:
        data = []

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
            target[2] += np.random.uniform(0, translation_noise)
            target[3] += np.random.uniform(-rotation_noise, rotation_noise)

            goal_pos = target[:3]
            goal_orientation = [-np.pi, 0, target[3]]

            env.env_method("set_goal", indices=i, goal_pos=goal_pos, goal_orientation=goal_orientation)

        while not np.all(dones):
            expert_actions_to_bottleneck = expert_policy_to_bottleneck(env, bottleneck)
            expert_actions_to_target = expert_policy_to_target(env)

            active_envs = np.logical_not(dones)
            active_envs_indices = [i for i in range(env.num_envs) if active_envs[i]]

            for i in active_envs_indices:
                data.append({"image": obss[i], "action": expert_actions_to_bottleneck[i], "endpoint_height": env.env_method("get_position", indices=i)[2]})

            # Get expert actions to target with current observation
            active_expert_actions_to_target = [expert_actions_to_target[i] if active_envs[i] else None for i in range(env.num_envs)]
            env.step_async(active_expert_actions_to_target)
            obss, rewards, active_dones, _ = env.step_wait()

            # Update reward for each env
            for i in range(env.num_envs):
                if active_envs[i]:
                    total_return[i] += rewards[i]
                    steps += 1

            # Update done for each env
            for i in range(env.num_envs):
                if active_envs[i]:
                    dones[i] = active_dones[i]
                
            # If file doesn't exist or is empty, create new one
            if not os.path.exists(logdir + output_file) or os.stat(logdir + output_file).st_size == 0:
                with open(logdir + output_file, "wb") as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            # Else append to existing file
            else:
                with open(logdir + output_file, "rb") as f:
                    existing_data = pickle.load(f)
                with open(logdir + output_file, "wb") as f:
                    pickle.dump(existing_data + data, f, pickle.HIGHEST_PROTOCOL)

        
        # Append distances to goal and orientation z diffs
        for info in env.reset_infos:
            distances_to_goal.append(info["final_distance"])
            orientation_diffs_z.append(info["final_orientation"])

        # Append returns
        for i in range(env.num_envs):
            returns[i].append(total_return[i])
        
        # Get amount of data in file
        with open(logdir + output_file, "rb") as f:
            amount_of_data_collected = len(pickle.load(f))
            print("amount of data:", amount_of_data_collected)

    print("done collecting data")
    # print mean and std of returns, distances to goal and orientation z diffs
    print("mean return:", np.mean(total_return))

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