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
import matplotlib.pyplot as plt

from tqdm import trange

def expert_policy(env):
    # agent_position = env.agent.get_position()
    # agent_orientation = env.agent.get_orientation()
    # goal_position = env.goal_pos
    # goal_orientation = env.goal_orientation

    # agents = env.get_attr("agent")
    agent_positions = env.env_method("get_agent_position")
    agent_orientations = env.env_method("get_agent_orientation")
    goal_positions = env.get_attr("goal_pos")
    goal_orientations = env.get_attr("goal_orientation")

    x_diffs = []
    y_diffs = []
    z_diffs = []
    orientation_diffs_z = []
    actions = []

    for i in range(len(agent_positions)):
        x_diffs.append(goal_positions[i][0] - agent_positions[i][0])
        y_diffs.append(goal_positions[i][1] - agent_positions[i][1])
        z_diffs.append(goal_positions[i][2] - agent_positions[i][2])
        orientation_diff_z = goal_orientations[i][2] - agent_orientations[i][2]
        if orientation_diff_z < -np.pi:
            orientation_diff_z += 2 * np.pi
        elif orientation_diff_z > np.pi:
            orientation_diff_z -= 2 * np.pi
        orientation_diffs_z.append(orientation_diff_z)

    for i in range(len(x_diffs)):
        action = np.array([x_diffs[i], y_diffs[i], z_diffs[i], orientation_diffs_z[i]], dtype=np.float32)
        actions.append(action)

    # x_diff = goal_position[0] - agent_position[0]
    # y_diff = goal_position[1] - agent_position[1]
    # z_diff = goal_position[2] - agent_position[2]
    # orientation_diff_z = goal_orientation[2] - agent_orientation[2]

    # if orientation_diff_z < -np.pi:
    #     orientation_diff_z += 2 * np.pi
    # elif orientation_diff_z > np.pi:
    #     orientation_diff_z -= 2 * np.pi

    return actions

def collect_data(scene_file_name, bottleneck, num_of_samples=100):

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
    # env = Monitor(gym.make("RobotEnv-v2", file_name=scene_file_name, bottleneck=bottleneck), logdir)

    env = make_vec_env("RobotEnv-v2",
                        n_envs=16,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs=dict(file_name=scene_file_name, bottleneck=bottleneck))

    returns = [[] for _ in range(env.num_envs)]
    observations = []
    actions = []
    distances_to_goal = []
    orientation_z_diffs = []

    while len(observations) < num_of_samples:

        obss = env.reset()
        
        dones = np.zeros((env.num_envs,), dtype=bool)
        total_return = np.zeros((env.num_envs,))
        steps = 0

        while not np.all(dones):
            expert_actions = expert_policy(env)

            # Only step in not done environments
            active_envs = np.logical_not(dones)
            active_actions = [action if active else None for action, active in zip(expert_actions, active_envs)]
            env.step_async(active_actions)
            active_obss, active_rewards, active_dones, _ = env.step_wait()

            # Filter out None actions and observations
            active_actions = [action for action in active_actions if action is not None]
            active_obss = [obs for obs, action in zip(active_obss, active_actions) if action is not None]

            observations += active_obss
            actions += active_actions

            # Update return for each environment
            for i in np.where(active_envs)[0]:
                total_return[i] += active_rewards[i]

            # Update the done status for active environments
            dones = np.logical_or(dones, active_dones)
            steps += np.sum(active_envs)

            # if len(observations) >= num_of_samples:
            #     break

            if(len(observations) % 100 == 0):
                print("amount of data:", len(observations))
                print(len(observations) / num_of_samples * 100, "%")

        # Append returns for each environment separately
        for i, ret in enumerate(total_return):
            returns[i].append(ret)

        # Append final distance and orientation to distance and orientation lists
        for info in env.reset_infos:
            distances_to_goal.append(info["final_distance"])
            orientation_z_diffs.append(info["final_orientation"])

    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))
    print("mean distance to goal", np.mean(distances_to_goal))
    print("std of distance to goal", np.std(distances_to_goal))
    print("mean orientation z diff", np.mean(orientation_z_diffs))
    print("std of orientation z diff", np.std(orientation_z_diffs))

    scene = scene_file_name[:-4]

    expert_data = {"observations": np.array(observations),
                     "actions": np.array(actions)}

    output_file = scene + "_expert_data_" + str(num_of_samples) + ".pkl"
    with open(logdir + output_file, "wb") as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    # Save the above results to a txt file with scene
    with open(logdir + scene + "_results.txt", "w") as f:
        f.write("mean return: " + str(np.mean(returns)) + "\n")
        f.write("std of return: " + str(np.std(returns)) + "\n")
        f.write("mean distance to goal: " + str(np.mean(distances_to_goal)) + "\n")
        f.write("std of distance to goal: " + str(np.std(distances_to_goal)) + "\n")
        f.write("mean orientation z diff: " + str(np.mean(orientation_z_diffs)) + "\n")
        f.write("std of orientation z diff: " + str(np.std(orientation_z_diffs)) + "\n")
