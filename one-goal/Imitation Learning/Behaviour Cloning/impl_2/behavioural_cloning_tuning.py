import pickle
import numpy as np
import tensorflow as tf
import gymnasium as gym
import pprint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from tqdm import trange
import os
import keras_tuner
from tensorflow.python.keras import backend as K
print(K._get_available_gpus())

SPLIT_RATIO = 0.8
MODEL_INDEX = 1

POSITION_COEFFICIENT = 0.8
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

def get_data(data_file, val_file, test_file):
    train_data = load_data(data_file)
    obs_train = np.array(train_data['observations'])
    action_train = np.array(train_data['actions'])

    # val_data = load_data(val_file)
    # obs_val = np.array(val_data['observations'])
    # action_val = np.array(val_data['actions'])

    # test_data = load_data(test_file)
    # obs_test = np.array(test_data['observations'])
    # action_test = np.array(test_data['actions'])

    obs_train, obs_val, action_train, action_val = train_test_split(obs_train, action_train, test_size=0.2, random_state=42)
    obs_train, obs_test, action_train, action_test = train_test_split(obs_train, action_train, test_size=0.2, random_state=42)

    obs_train = np.transpose(obs_train, (0, 2, 3, 1))
    obs_test = np.transpose(obs_test, (0, 2, 3, 1))
    obs_val = np.transpose(obs_val, (0, 2, 3, 1))

    input_shape = obs_train.shape[1:]
    output_nodes = 4

    print("Input shape: ", input_shape)
    print("Obs train shape: ", obs_train.shape)
    print("Obs test shape: ", obs_test.shape)
    print("Obs val shape: ", obs_val.shape)

    # Normalise obs_train, obs_test and obs_val
    obs_train = obs_train / 255.0
    obs_test = obs_test / 255.0
    obs_val = obs_val / 255.0
    
    print("before:", action_train)
    # for each action, divide the last value by np.pi
    action_train[:, 3] = action_train[:, 3] / (np.pi)
    print("after:", action_train)
    action_test[:, 3] = action_test[:, 3] / (np.pi)
    action_val[:, 3] = action_val[:, 3] / (np.pi)

    return obs_train, obs_val, obs_test, action_train, action_val, action_test, input_shape, output_nodes

class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        # Create model
        print("Creating model...")
        input_shape = (64, 64, 3)
        output_nodes = 4

        model = Sequential()

        number_of_layers = hp.Int('num_layers', 3, 4, default=4)
        net_arch = []

        for i in range(number_of_layers):
            net_arch.append(hp.Choice('units_' + str(i), [64, 96, 128, 256, 512]))

        model.add(Conv2D(net_arch[0], (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        dropout_rate = hp.Choice('dropout_rate', values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        print(net_arch)

        for layer_size in net_arch[1:]:
            model.add(Conv2D(layer_size, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        
        model.add(Flatten())

        number_of_dense_layers = hp.Int('num_dense_layers', 2, 4)
        for i in range(number_of_dense_layers):
            dense_layer_size = hp.Int('dense_layer_size_' + str(i), min_value=150, max_value=500, step=50)
            model.add(Dense(dense_layer_size, activation='relu'))
            model.add(Dropout(dropout_rate))

        model.add(Dense(output_nodes, activation='tanh'))

        # Compile the model with a loss function and optimizer
        print("Compiling the model...")

        # hyperparameter for optimiser and lr scheduler
        learning_rate = hp.Float("lr", min_value=0.001, max_value=0.1, sampling="log")
        
        # Scheduler or static learning rate
        # has_lr_schedule = hp.Boolean("has_lr_schedule")
        # with hp.conditional_scope("has_lr_schedule", True):
        #     if has_lr_schedule:
        #         # lr scheduler
        #         lr_scheduler = hp.Choice('lr_scheduler', values=['ExponentialDecay', 'CosineDecay'])
        #         decay_step = hp.Int("decay_step", min_value=1000, max_value=10000)

        #         with hp.conditional_scope('lr_scheduler', ['ExponentialDecay']):
        #             if lr_scheduler == 'ExponentialDecay':
        #                 decay_rate = hp.Float("decay_rate", min_value=0.9, max_value=0.99)
        #                 lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #                     initial_learning_rate=learning_rate,
        #                     decay_steps=decay_step,
        #                     decay_rate=decay_rate)
        #         with hp.conditional_scope('lr_scheduler', ['CosineDecay']):
        #             if lr_scheduler == 'CosineDecay':
        #                 lr_schedule = tf.keras.experimental.CosineDecay(
        #                     initial_learning_rate=learning_rate,
        #                     decay_steps=decay_step)
        #         optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # with hp.conditional_scope("has_lr_schedule", False):
        #     if not has_lr_schedule:
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        POSITION_COEFFICIENT = hp.Float("position_coefficient", min_value=0.01, max_value=1.0, step=0.1)
        ORIENTATION_COEFFICIENT = hp.Float("orientation_coefficient", min_value=0.01, max_value=1.0, step=0.1)

        # compile with adam optimiser and lr scheduler
        model.compile(loss=custom_loss, optimizer=optimizer) 

        print(model.summary())
        print("Model created")

        pprint.pprint({k: hp.get(k) if hp.is_active(k) else None for k in hp._hps})

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [64, 128, 256, 512]),
            **kwargs,
        )

def tune_model():
    logdir = "/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/"
    data_file = logdir + "expert_data/pitcher_scene_expert_data_100000.pkl"
    val_file = logdir + "expert_data/twist_shape_scene_expert_data_20000.pkl"
    test_file = logdir + "expert_data/twist_shape_scene_expert_data_10000.pkl"

    obs_train, obs_val, obs_test, action_train, action_val, action_test, input_shape, output_nodes = get_data(data_file, val_file, test_file)

    tuner = keras_tuner.BayesianOptimization(
        MyHyperModel(),
        objective='val_loss',
        max_trials=10000,
        executions_per_trial=1,
        max_consecutive_failed_trials=10,
        directory=logdir,
        project_name='bc-hyperparameters5')

    tuner.search_space_summary(extended=True)

    tuner.search(obs_train, action_train,
                epochs=200,
                validation_data=(obs_val, action_val),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=5),
                            tf.keras.callbacks.TensorBoard(log_dir=logdir + "bc-hyperparameters5/tensorboard"),
                            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)])

    tuner.results_summary()
    
    print("Best hyperparameters: ", tuner.get_best_hyperparameters()[0])


tune_model()