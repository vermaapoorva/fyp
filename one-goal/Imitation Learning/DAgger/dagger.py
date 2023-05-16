import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import robot_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from tqdm import trange
import os
from tensorflow.python.keras import backend as K
print(K._get_available_gpus())

SPLIT_RATIO = 0.8
MODEL_INDEX = 1
NUM_EPOCHS = 30
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
NUM_OF_DAGGER_ITERATIONS = 10
NUM_OF_ROLLOUTS_PER_DAGGER_ITERATION = 20
INPUT_SHAPE = (64, 64, 3)
OUTPUT_NODES = 4
DATA_FILE = f"expert_data_{MODEL_INDEX}.pkl"

POSITION_COEFFICIENT = 1
ORIENTATION_COEFFICIENT = 0.01

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def custom_loss(action_true, action_pred):

    pos_action_true = action_true[:, :3]
    pos_action_pred = action_pred[:, :3]
    ori_action_true = action_true[:, 3]
    ori_action_pred = action_pred[:, 3]

    pos_loss = tf.keras.losses.MSE(pos_action_true, pos_action_pred)
    ori_loss = tf.keras.losses.MSE(ori_action_true, ori_action_pred)

    loss = POSITION_COEFFICIENT * pos_loss + ORIENTATION_COEFFICIENT * ori_loss
    
    return loss

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

def create_model():
    # Create model
    print("Creating model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Conv2D(48, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(200, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(200, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(50, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(OUTPUT_NODES)
    ])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

    # Compile the model with a loss function and optimizer
    print("Compiling the model...")
    model.compile(loss=custom_loss, optimizer='adam')

    # Save the model
    print("Saving the trained model...")
    model.save("models/model_" + str(MODEL_INDEX) + ".h5")

def train_model(env):

    # Delete old data file
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    for i in trange(NUM_OF_DAGGER_ITERATIONS):

        # Load the model
        print("Loading the model...")
        model = load_model("models/model_" + str(MODEL_INDEX) + ".h5", custom_objects={'custom_loss': custom_loss})

        ##########################################################################################
        ################################### Get expert data ######################################
        ##########################################################################################
        
        returns = []
        new_observations = []
        new_expert_actions = []
        distances_to_goal = []
        orientation_z_diffs = []

        for i in trange(NUM_OF_ROLLOUTS_PER_DAGGER_ITERATION):
            obs, _ = env.reset()
            done = False
            total_return = 0
            steps = 0

            while not done:
                # Get expert action for current state
                expert_action = expert_policy(env)
                new_observations.append(obs)
                new_expert_actions.append(expert_action)

                # Get model action for current state
                obs = np.expand_dims(obs, axis=0).transpose(0, 2, 3, 1)
                action = model.predict(obs, verbose=0)[0]

                # Take model's action in environment
                obs, reward, done, truncated, info = env.step(action)
                total_return += reward
                steps += 1

            print(f"Rollout {i} finished\nReturn: {total_return}\nSteps: {steps}\nDistance to goal: {env.get_distance_to_goal()}\nOrientation z diff: {env.get_orientation_diff_z()}\n")
            returns.append(total_return)
            distances_to_goal.append(env.get_distance_to_goal())
            orientation_z_diffs.append(env.get_orientation_diff_z())

        print("mean return", np.mean(returns))
        print("mean distance to goal", np.mean(distances_to_goal))
        print("mean orientation z diff", np.mean(orientation_z_diffs))

        ##########################################################################################
        ################################# Save new expert data ###################################
        ##########################################################################################

        new_observations = np.array(new_observations)
        new_expert_actions = np.array(new_expert_actions)

        # If expert data file does not exist, create it
        if not os.path.exists(DATA_FILE):
            print("Creating new expert data file...")
            expert_data = {'observations': new_observations, 'actions': new_expert_actions}
            with open(DATA_FILE, "wb") as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

        # If expert data file exists, load it and append new data
        else:
            print("Appending new expert data to existing file...")
            expert_data = load_data(DATA_FILE)
            expert_data['observations'] = np.concatenate((expert_data['observations'], new_observations))
            expert_data['actions'] = np.concatenate((expert_data['actions'], new_expert_actions))
            with open(DATA_FILE, "wb") as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)
        
        print("Num of expert data obs and actions: ", expert_data['observations'].shape, expert_data['actions'].shape)

        ##########################################################################################
        ################################ Load all expert data ####################################
        ##########################################################################################

        ### Load data
        data = load_data(DATA_FILE)
        obs_data = np.array(data['observations'])
        action_data = np.array(data['actions'])

        num_of_samples = obs_data.shape[0]
        obs_data, action_data = shuffle(obs_data, action_data, random_state=0)

        obs_train, obs_test, action_train, action_test = train_test_split(obs_data, action_data, test_size=1-SPLIT_RATIO, random_state=0)
        obs_train, obs_val, action_train, action_val = train_test_split(obs_train, action_train, test_size=1-SPLIT_RATIO, random_state=0)

        obs_train = np.transpose(obs_train, (0, 2, 3, 1))
        obs_test = np.transpose(obs_test, (0, 2, 3, 1))
        obs_val = np.transpose(obs_val, (0, 2, 3, 1))

        input_shape = obs_train.shape[1:]
        output_nodes = 4

        ##########################################################################################
        ##################################### Train model ########################################
        ##########################################################################################
        
        # Train the model
        print("Training the model...")
        history = model.fit(obs_train, action_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, validation_data=(obs_val, action_val))

        # Evaluate the trained model on a test dataset
        print("Evaluating the trained model on a test dataset...")
        mse = model.evaluate(obs_test, action_test, verbose=1)
        print("Mean squared error: ", mse)

        # Save the trained model
        print("Saving the trained model...")
        model.save("models/model_" + str(MODEL_INDEX) + ".h5")

def run_model(env):
    print("Running model")

    model = load_model("models/model_" + str(MODEL_INDEX) + ".h5", custom_objects={'custom_loss': custom_loss})

    # Evaluate the trained model
    print("Calculating accuracy of model")
    returns = []
    distances_to_goal = []
    orientation_z_diffs = []
    steps_list = []

    for i in trange(15):

        obs, _ = env.reset()
        done = False
        total_return = 0
        steps = 0

        while not done:
            obs = np.expand_dims(obs, axis=0).transpose(0, 2, 3, 1)
            action = model.predict(obs, verbose=0)[0]
            obs, reward, done, truncated, info = env.step(action)
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

if __name__ == "__main__":
    create_model()
    env = Monitor(gym.make("RobotEnv-v2"), "logs/")
    train_model(env)
    run_model(env)  