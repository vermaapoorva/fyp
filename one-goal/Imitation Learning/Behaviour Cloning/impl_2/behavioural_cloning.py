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
# MODEL_INDEX = 1
# NUM_EPOCHS = 500
# BATCH_SIZE = 32
# DROPOUT_RATE = 0.2
# LEARNING_RATE = 0.001
# DATA_FILE = "expert_data_500.pkl"

POSITION_COEFFICIENT = 1
ORIENTATION_COEFFICIENT = 0.8

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

def train_model(model_index,
                num_epochs,
                batch_size,
                dropout_rate,
                learning_rate,
                data_file):
    
    print("Training model")

    # POSITION_COEFFICIENT = position_coefficient
    # ORIENTATION_COEFFICIENT = orientation_coefficient
    MODEL_INDEX = model_index
    NUM_EPOCHS = num_epochs
    BATCH_SIZE = batch_size
    DROPOUT_RATE = dropout_rate
    LEARNING_RATE = learning_rate
    
    data = load_data(data_file)
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

    print("Input shape: ", input_shape)
    print("Obs train shape: ", obs_train.shape)
    print("Obs test shape: ", obs_test.shape)
    print("Obs val shape: ", obs_val.shape)

    # Create model
    print("Creating model...")

    # NatureCNN architecture used as default feature extractor PPO
    # nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
    # nn.ReLU(),
    # nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
    # nn.ReLU(),
    # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
    # nn.ReLU(),
    # nn.Flatten(),


    model = Sequential([
        # Conv2D(32, (8, 8), strides=(4, 4), padding='same', activation='relu', input_shape=input_shape),
        # Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu'),
        # Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
        Dense(output_nodes)
    ])

    earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Compile the model with a loss function and optimizer
    print("Compiling the model...")
    model.compile(loss=custom_loss, optimizer='adam')

    # Train the model
    print("Training the model...")
    history = model.fit(obs_train, action_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, callbacks=[earlyStoppingCallback, learningRateSchedulerCallback], verbose=1, validation_data=(obs_val, action_val))

    # Save the training history
    print("Saving the training history...")
    os.makedirs('training_history', exist_ok=True)
    with open('training_history/model_' + str(MODEL_INDEX) + ".pkl", 'wb') as history_file:
        pickle.dump(history.history, history_file)

    # Plot the training and validation loss
    print("Plotting the training and validation loss...")
    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # Save plot
    plt.savefig('loss_' + str(MODEL_INDEX) + '.png')
    plt.clf()

    # Evaluate the trained model on a test dataset
    print("Evaluating the trained model on a test dataset...")
    mse = model.evaluate(obs_test, action_test, verbose=1)
    print("Mean squared error: ", mse)

    # Save the trained model
    print("Saving the trained model...")
    model.save("models/model_" + str(MODEL_INDEX) + ".h5")

def run_model(model_index, env):
    print("Running model")
    MODEL_INDEX = model_index

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
    print("=====================================================================")

    # Appending the accuracies of the model to a text file
    with open("model_accuracies.txt", "a") as f:
        f.write("Model " + str(MODEL_INDEX) + "\n")
        f.write("Steps: " + str(np.mean(steps_list)) + "\n")
        f.write("Distance: " + str(np.mean(distances_to_goal)) + "\n")
        f.write("Orientation: " + str(np.mean(orientation_z_diffs)) + "\n")
        f.write("=====================================================================\n")