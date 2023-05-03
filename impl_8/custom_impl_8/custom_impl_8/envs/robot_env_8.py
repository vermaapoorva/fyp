import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np
from PIL import Image

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object
from os.path import dirname, join, abspath

import time
SCENE_FILE = join(dirname(abspath(__file__)), 'impl_8_scene.ttt')

class RobotEnv8(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, headless=True, image_size=64, sleep=0):
        super(RobotEnv8, self).__init__()
        print("init")
        self.image_size = image_size
        self.sleep = sleep
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -np.pi]),
                                       high=np.array([1, 1, 1, np.pi]),
                                       shape=(4,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8)
        
        # self.observation_space = spaces.Dict({"camera_image": spaces.Box(low=0, high=255,
        #                                     shape=(3, self.image_size, self.image_size), dtype=np.uint8),
        #                                       "goal_image": spaces.Box(low=0, high=255,
        #                                     shape=(3, self.image_size, self.image_size), dtype=np.uint8)})

        self.done = False
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation

        self.agent = VisionSensor("camera")
        self.target = Object("target")
        self.initial_target_pos = self.target.get_position()
        self.step_number = 0
        self.goal_pos = self.get_random_goal_pos()
        self.goal_theta, self.goal_quaternion = self.get_random_goal_quaterunion()
        self.goal_image = self._get_goal_image()

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_quaternion(self.get_random_agent_quaterunion())

    def _get_goal_image(self):
        self.agent.set_position(self.goal_pos)
        goal_image = self._get_current_image()
        img = Image.fromarray(goal_image.transpose(1, 2, 0))
        img.save("goal_image_" + str(self.goal_pos) + "_" + str(self.goal_quaternion) + ".jpg")
        return goal_image

    def _get_state(self):
        return self._get_current_image()

    def _get_current_image(self):
        image = self.agent.capture_rgb()
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        img = Image.fromarray(resized)
        if self.step_number % 100 == 0:
            img.save("image_" + str(self.step_number) + ".jpg")
        return resized.transpose(2, 0, 1)

    def step(self, action):

        self.pr.step()
        self.step_number += 1
        action_scale = 0.01
        action_x, action_y, action_z, action_theta = action

        # new_x, new_y, new_z = self.agent.get_position() + action_scale*[action_x, action_y, action_z]

        new_x = self.agent.get_position()[0] + action_scale * action_x
        new_y = self.agent.get_position()[1] + action_scale * action_y
        new_z = self.agent.get_position()[2] + action_scale * action_z

        # If within range, move
        if new_x > -0.25 and new_x < 0.25 and new_y > -0.25 and new_y < 0.25 and new_z > 0.8 and new_z < 2:
            self.agent.set_position([new_x, new_y, new_z])
        
        # Rotate
        self.agent.set_quaternion(self.get_quaternion(action_theta))

        tx, ty, tz = self.goal_pos
        reward = -(np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2 + (new_z - tz) ** 2) + np.abs(action_theta - self.goal_theta))

        done = False
        truncated = False
        if reward > -0.01:
            done = True
            reward = 200
        if self.step_number == 500:
            done = True
            truncated = True
            self.step_number = 0

        time.sleep(self.sleep)
        if done:
            time.sleep(self.sleep * 100)
        
        self.target.set_position(self.initial_target_pos)

        return self._get_state(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.goal_pos = self.get_random_goal_pos()
        self.goal_theta, self.goal_quaternion = self.get_random_goal_quaterunion()
        self.goal_image = self._get_goal_image()
        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_quaternion(self.get_random_agent_quaterunion())
        self.target.set_position(self.initial_target_pos)
        return self._get_state(), {}  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close (self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

    def get_random_agent_pos(self):
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.25, 0.25)
        z = np.random.uniform(1, 2)
        return [x, y, z]
    
    # get an orientation rotated just in z axis
    def get_random_agent_quaterunion(self):
        theta = np.random.uniform(-np.pi, np.pi)
        return self.get_quaternion(theta)

    def get_random_goal_pos(self):
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(1, 1.5)
        return [x, y, z]
    
    # get an orientation rotated just in z axis
    def get_random_goal_quaterunion(self):
        theta = np.random.uniform(-np.pi, np.pi)
        return theta, self.get_quaternion(theta)
    
    def get_quaternion(self, theta):
        # theta = 0
        x = 0
        y = 0
        z = np.cos(theta/2)
        w = -np.sin(theta/2)
        return [x, y, z, w]
    
    def get_distance_to_goal(self):
        x, y, z = self.agent.get_position()
        tx, ty, tz = self.goal_pos
        return np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
