import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils
from sklearn.utils import shuffle

SPLIT_RATIO = 0.8
MODEL_INDEX = 0

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def train_model(data_file):
    
    print("Training model")
    
    data = load_data(data_file)
    obs_data = np.array(data['observations'])
    action_data = np.array(data['actions'])

    num_of_samples = obs_data.shape[0]
    obs_data, action_data = shuffle(obs_data, action_data, random_state=0)
    split_index = int(num_of_samples * SPLIT_RATIO)

    obs_train = obs_data[:split_index]
    obs_test = obs_data[split_index:]
    action_train = action_data[:split_index]
    action_test = action_data[split_index:]

    obs_train = np.transpose(obs_train, (0, 2, 3, 1))
    obs_test = np.transpose(obs_test, (0, 2, 3, 1))

    input_shape = obs_train.shape[1:]
    output_nodes = 4

    print("input_shape: ", input_shape)
    print("obs_train_shape: ", obs_train.shape)
    print("obs_test shape: ", obs_test.shape)
    print("action_train_shape: ", action_train.shape)
    print("action_test.shape: ", action_test.shape)

    print("num of samples:", num_of_samples)

    # Create model
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(output_nodes)
    ])

    # Compile the model with a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')
    model.fit(obs_train, action_train, batch_size=512, epochs=30, verbose=1)
    score = model.evaluate(obs_test, action_test, verbose=1)

    model.save("models/model_" + str(MODEL_INDEX) + ".h5")

def run_model():
    print("Running model")


if __name__ == '__main__':
    data_file = "expert_data_50.pkl"
    train_model(data_file)
    run_model()
