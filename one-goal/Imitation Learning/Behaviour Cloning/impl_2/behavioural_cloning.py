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
from stable_baselines3.common.monitor import Monitor
from tqdm import trange
import os
from tensorflow.python.keras import backend as K
print(K._get_available_gpus())

SPLIT_RATIO = 0.8
# model_index = 1
# NUM_EPOCHS = 500
# BATCH_SIZE = 32
# DROPOUT_RATE = 0.2
# LEARNING_RATE = 0.001
# DENSE_ARCH = [100, 100]
# NET_ARCH = [32, 64, 128, 256]

POSITION_COEFFICIENT = 0.3
ORIENTATION_COEFFICIENT = 1

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

def create_model(input_shape, output_nodes, net_arch, dense_arch, dropout_rate, learning_rate):
    # Create model
    print("Creating model...")

    model = Sequential()

    model.add(Conv2D(net_arch[0], (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    for layer_size in net_arch[1:]:
        print(f"Adding conv layer with {layer_size} nodes")
        model.add(Conv2D(layer_size, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    model.add(Flatten())

    for layer_size in dense_arch:
        print(f"Adding dense layer with {layer_size} nodes")
        model.add(Dense(layer_size, activation='relu'))
        # model.add(Dropout(dropout_rate))

    model.add(Dense(output_nodes, activation='tanh'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model with a loss function and optimizer
    print("Compiling the model...")
    model.compile(loss=custom_loss, optimizer=optimizer)

    print(model.summary())
    print("Model created")

    return model

def train_model(model_index,
                num_epochs,
                batch_size,
                dropout_rate,
                learning_rate,
                net_arch,
                dense_arch,
                data_file,
                val_file,
                test_file,
                amount_of_data,
                pos_coeff,
                or_coeff):
    
    print("Training model")
    POSITION_COEFFICIENT = pos_coeff
    ORIENTATION_COEFFICIENT = or_coeff

    # print(f"Hyperparameters: epochs {num_epochs}, batch size {batch_size}, dropout rate {dropout_rate}, learning rate {learning_rate}, net arch {net_arch}, dense arch {dense_arch}")
    
    train_data = load_data(data_file)
    obs_data = np.array(train_data['observations'])
    action_data = np.array(train_data['actions'])

    # only train on x samples
    obs_data = obs_data[:amount_of_data]
    action_data = action_data[:amount_of_data]

    num_of_samples = obs_data.shape[0]
    print("Num samples: ", num_of_samples)
    obs_data, action_data = shuffle(obs_data, action_data, random_state=42)

    # change images in obs_data to channel last
    obs_data = np.transpose(obs_data, (0, 2, 3, 1))

    obs_train, obs_test, action_train, action_test = train_test_split(obs_data, action_data, test_size=1-SPLIT_RATIO, random_state=42)
    obs_train, obs_val, action_train, action_val = train_test_split(obs_train, action_train, test_size=1-SPLIT_RATIO, random_state=42)
   
    # get index of obs_train[0] in obs_data
    index = np.where(np.all(obs_data == obs_train[0], axis=(1,2,3)))
    print(f"index of obs: {index}")
    action = np.where(np.all(action_data == action_train[0], axis=1))
    print(f"index of action: {action}")

    input_shape = obs_train.shape[1:]
    output_nodes = 4
    
    print("before:", action_train)
    # for each action, divide the last value by np.pi
    action_train[:, 3] = action_train[:, 3] / (np.pi)
    print("after:", action_train)
    action_test[:, 3] = action_test[:, 3] / (np.pi)
    action_val[:, 3] = action_val[:, 3] / (np.pi)

    print("Input shape: ", input_shape)
    print("Obs train shape: ", obs_train.shape)
    print("Obs test shape: ", obs_test.shape)
    print("Obs val shape: ", obs_val.shape)

    model = create_model(input_shape, output_nodes, net_arch=net_arch, dense_arch=dense_arch, dropout_rate=dropout_rate, learning_rate=learning_rate)

    # earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, mode='min')
    # learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/tensorboard_logs/model_" + str(model_index))
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=1e-10)
    
    # Train the model
    print("Training the model...")
    history = model.fit(obs_train, action_train, batch_size=batch_size, epochs=num_epochs, verbose=1, callbacks=[tensorboard_callback, reduce_lr], validation_data=(obs_val, action_val))

    # Save the training history create folder if it doesn't exist
    print("Saving the training history...")
    if not os.path.exists('/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/training_history'):
        os.makedirs('/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/training_history')
    
    with open('/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/training_history/model_' + str(model_index) + ".pkl", 'wb') as history_file:
        pickle.dump(history.history, history_file)

    # Plot the training and validation loss
    print("Plotting the training and validation loss...")
    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # Save plot to losses folder with model_index in name
    plt.savefig('losses/loss' + str(model_index) + '.png')
    plt.clf()

    # Evaluate the trained model on a test dataset
    print("Evaluating the trained model on a test dataset...")
    mse = model.evaluate(obs_test, action_test, verbose=1)
    print("Mean squared error: ", mse)

    # Save the trained model
    print("Saving the trained model...")
    model.save("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/models/" + str(model_index) + "_model.h5")

    return mse

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

def run_model(model_index, env, num_of_runs=20, hp_index=None, scene_index=None):
    print("Running model")

    model = load_model("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/models/" + str(model_index) + "_model.h5", custom_objects={'custom_loss': custom_loss})

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
            obs = np.expand_dims(obs, axis=0).transpose(0, 2, 3, 1)
            # normalise obs
            obs = obs / 255.0
            expert_action = expert_policy(env)
            action = model.predict(obs, verbose=0)[0]
            # print(f"Model action before: {action}")
            # print(f"Expert action: {expert_action}")
            # multiply last action by pi to get back to radians
            action[3] = action[3] * np.pi
            # print(f"Model action after: {action}")
            obs, reward, done, truncated, info = env.step(expert_action)
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
    print("=====================================================================")

    # Create model accuracies folder if it doesn't exist
    os.makedirs('model_accuracies', exist_ok=True)

    # Appending the accuracies of the model to a text file
    with open("model_accuracies/random_search_tuning.txt", "a") as f:
        f.write("Model " + str(model_index) + "\n")
        f.write("Mean return: " + str(np.mean(returns)) + "\n")
        f.write("Mean steps: " + str(np.mean(steps_list)) + "\n")
        f.write("Mean distance: " + str(np.mean(distances_to_goal)) + "\n")
        f.write("Mean orientation: " + str(np.mean(orientation_z_diffs)) + "\n")
        f.write("=====================================================================\n")

    # # Save the accuracies of the model
    # print("Saving the accuracies of the model...")
    # if hp_index is not None and scene_index is not None:
    #     with open('model_accuracies/hp_tuning_net_arch_results.pkl', 'ab') as accuracies_file:
    #         # write the accuracies to a file
    #         pickle.dump([hp_index, scene_index, np.mean(steps_list), np.mean(distances_to_goal), np.mean(orientation_z_diffs)], accuracies_file)

    print("Appending the accuracies to pkl file...")
    accuracies = []
    if os.path.exists('model_accuracies/random_search_tuning.pkl'):
        with open('model_accuracies/random_search_tuning.pkl', 'rb') as accuracies_file:
            accuracies = pickle.load(accuracies_file)
    
    accuracies.append([hp_index, scene_index, np.mean(returns), np.mean(steps_list), np.mean(distances_to_goal), np.mean(orientation_z_diffs)])
    with open('model_accuracies/random_search_tuning.pkl', 'wb') as accuracies_file:
        pickle.dump(accuracies, accuracies_file)

    return np.mean(returns), np.mean(distances_to_goal), np.mean(orientation_z_diffs)

def get_number_of_parameters(model_index):
    model_index = model_index
    model = load_model("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/models/" + str(model_index) + "_model.h5", custom_objects={'custom_loss': custom_loss})

    return model.count_params()
