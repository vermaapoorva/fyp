# import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import cv2

from sklearn.model_selection import train_test_split
import numpy as np

from PIL import Image
from os.path import dirname, join, abspath

import matplotlib.pyplot as plt

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Camera

import time

SCENE_FILE = join(dirname(abspath(__file__)), 'baseline_scene.ttt')

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

#################################################################################################
################################   SETTING UP THE ENVIRONMENT   #################################
#################################################################################################

class RobotEnvBaseline():

    def __init__(self, headless=True, image_size=64, sleep=0):
        print("init")
        self.image_size = image_size
        self.sleep = sleep
        
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation

        self.done = False
        self.completed = False
        self.agent = VisionSensor("camera")
        self.target = Object("target")
        self.initial_agent_pos = self.agent.get_position()
        self.initial_target_pos = self.target.get_position()
        self.goal_pos = [0, 0, 1]
        self.step_number = 0
        
        self.agent.set_position(self.get_random_agent_pos())

    def _get_state(self):
        # Return state containing image
        image = self.agent.capture_rgb()
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        return resized

    # If action is given then testing, else collect data
    def step(self, action=None):

        self.pr.step()
        self.step_number += 1
        action_scale = 0.01
        testing = True
        if action is None:
            # If no action is given, use the model to predict the next action
            action = self.get_next_action()
            testing = False

        new_x, new_y, new_z = self.agent.get_position() + action_scale*action
        # If out of bounds, done = True
        if new_x < -0.25 or new_x > 0.25 or new_y < -0.25 or new_y > 0.25 or new_z < 0.8 or new_z > 2:
            print("Out of bounds")
            self.done = True

        # If reached height of target, done = True        
        if self.distance_to_goal() < 0.01:
            print("Reached target")
            self.done = True
            self.completed = True

        # If testing and max steps reached, done = True
        if testing and self.step_number == 500:
            print("Max steps reached")
            self.done = True
            self.step_number = 0
            self.completed = True

        self.agent.set_position([new_x, new_y, new_z])
        # print("distance to goal: ", self.distance_to_goal())

        time.sleep(self.sleep)
        if self.done:
            time.sleep(self.sleep * 100)
        
        return self._get_state(), self.done, action, self.completed
    
    def get_next_action(self):
        T = self.predict_T()
        T = T / np.linalg.norm(T)
        return T

    def predict_T(self):
        curr_x, curr_y, curr_z = self.agent.get_position()
        T = np.array([self.goal_pos[0] - curr_x, self.goal_pos[1] - curr_y, self.goal_pos[2] - curr_z])
        return T

    def distance_to_goal(self):
        dist = np.linalg.norm(np.array(self.agent.get_position()) - np.array(self.goal_pos))
        return dist

    def reset(self):
        random_agent_pos = self.get_random_agent_pos()
        self.agent.set_position(random_agent_pos)
        self.target.set_position(self.initial_target_pos)
        self.done = False
        self.completed = False
        self.step_number = 0
        return self._get_state()

    def close(self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

    def get_random_agent_pos(self):
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.25, 0.25)
        z = np.random.uniform(1, 2)
        return [x, y, z]
    
#################################################################################################
####################################   TRAINING THE MODEL   #####################################
#################################################################################################

checkpoint_path = "model_1_checkpoint.ckpt"

def collect_data():
    env = RobotEnvBaseline(headless=True, image_size=64)

    # Collect training data
    x_train = []
    y_train = []
    completed = []
    while len(x_train) < 100000:
        print("Current len of x_train: ", len(x_train))
        state = env.reset()
        state, done, action, complete = env.step()
        while not done:
            x_train.append(state)
            state, done, action, complete = env.step()
            y_train.append(action)
            completed.append(complete)

    print("Training data collected")
    print("Number of training examples: ", len(x_train))
    print("Number of completed episodes: ", sum(completed))    

    # Make training data numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Save training data to file as numpy arrays
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)

    # Save x_train as images
    # for i in range(len(x_train)):
    #   if i % 100 == 0:
    #     img = Image.fromarray(x_train[i], 'RGB')
    #     img.save("x_train_" + str(i) + ".jpg")

def create_model():
    # Define the input shape of the images
    input_shape = (64, 64, 3)

    # Define the number of output nodes (3 for a 3x1 column matrix)
    output_nodes = 3

    # Define the neural network architecture
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_nodes)
    ])

    # Compile the model with a loss function and optimizer
    model.compile(loss='mse', optimizer='adam')

    return model

def train_model():

    model = create_model()
    
    # Load the dataset
    x_train = np.load('x_train.npy') # input images
    y_train = np.load('y_train.npy') # output 3x1 column matrices

    # Preprocess the dataset
    x_train = x_train.astype('float32') / 255

    # Split the dataset into training, validation and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    
    # Train the neural network on the dataset
    history = model.fit(x_train, y_train, epochs=100, batch_size=512, validation_data=(x_val, y_val))

    # Plot the training and validation loss
    plt.plot(history.history['loss'][2:])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    # Save plot
    plt.savefig('loss_with_100000.png')

    # Evaluate the trained model on a test dataset
    print("Evaluating the trained model on a test dataset...")
    mse = model.evaluate(x_test, y_test)
    print("Mean squared error: ", mse)

    # Save the trained model for future use
    model.save('model.h5')

def evaluate_model():
    # # Load the trained model
    # print("Loading model")
    model = create_model()
    env = RobotEnvBaseline(headless=True, image_size=64)

    # print("Loading data")
    # # Load the dataset                                                                                                                                                                                
    # x_train = np.load('x_train.npy') # input images                                                                                                                                                   
    # y_train = np.load('y_train.npy') # output 3x1 column matrices                                                                                                                                      
    # # Preprocess the dataset                                                                                                                                                                      
    # x_train = x_train.astype('float32') / 255

    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # print("Evaluating model")
    # loss, acc = model.evaluate(x_test, y_test, verbose=2)
    # print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    # Loads the weights
    model.load_weights(checkpoint_path)

    # # Re-evaluate the model
    # loss, acc = model.evaluate(x_test, y_test, verbose=2)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    # Evaluate the trained model
    print("Calculating accuracy of model")
    accuracy = 0
    dists_when_reached_target = np.array([])
    while len(dists_when_reached_target) < 100:
        state = env.reset()
        done = False
        completed = False
        while not done:
            state = np.expand_dims(state, axis=0)
            action = model.predict(state)
            state, done, _, completed = env.step(action[0])

        if completed:
            print("Target reached")
            dist = env.distance_to_goal()
            dists_when_reached_target = np.append(dists_when_reached_target, dist)

    print("dists_when_reached_target: ", dists_when_reached_target)
    print("num of successful runs: ", dists_when_reached_target.shape)
    accuracy = np.mean(dists_when_reached_target)
    print("Average distance to target: ", accuracy)

# collect_data()
train_model()
# evaluate_model()
