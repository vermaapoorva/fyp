import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym

import robot_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import numpy as np
from math import pi
from fractions import Fraction

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


def collect_data(num_rollouts=20):

    logdir = "logs/"
    env = Monitor(gym.make("RobotEnv-v0"), "logs/")

    returns = []
    observations = []
    actions = []
    distances_to_goal = []
    orientation_z_diffs = []

    for i in trange(num_rollouts):

        print("iter", i)
        obs, _ = env.reset()
        done = False
        total_return = 0
        steps = 0

        while not done:
            expert_action = expert_policy(env)
            observations.append(obs)
            actions.append(expert_action)
            obs, reward, done, truncated, info = env.step(expert_action)
            total_return += reward
            steps += 1
            if done:
                break

        returns.append(total_return)
        distances_to_goal.append(env.get_distance_to_goal())
        orientation_z_diffs.append(env.get_orientation_diff_z())

    print("returns", returns)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))
    print("distances to goal", distances_to_goal)
    print("mean distance to goal", np.mean(distances_to_goal))
    print("std of distance to goal", np.std(distances_to_goal))
    print("orientation z diffs", orientation_z_diffs)
    print("mean orientation z diff", np.mean(orientation_z_diffs))
    print("std of orientation z diff", np.std(orientation_z_diffs))

    expert_data = {"observations": np.array(observations),
                     "actions": np.array(actions)}

    output_file = "expert_data_" + str(num_rollouts) + ".pkl"
    with open(output_file, "wb") as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    collect_data(50)
