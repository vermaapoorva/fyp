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
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -np.pi, -np.pi, -np.pi]),
                                       high=np.array([1, 1, 1, np.pi, np.pi, np.pi]),
                                       shape=(6,),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8)
        
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation
        
        self.step_number = 0
        self.done = False

        self.agent = VisionSensor("camera")
        self.goal_camera = VisionSensor("goal_camera")
        self.target = Object("target")
        self.initial_target_pos = self.target.get_position()

        self.goal_pos = [0, 0.175, 1]
        self.goal_orientation = [np.pi * 8/9, -np.pi/12, np.pi/9]
        self.goal_camera.set_position(self.goal_pos)
        self.goal_camera.set_orientation(self.goal_orientation)

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())

    def _get_state(self):
        current = self._get_current_image(self.agent)
        return current.transpose(2, 0, 1)

    def _get_current_image(self, camera):
        self.pr.step()
        image = camera.capture_rgb()
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        return resized

    def step(self, action):

        self.pr.step()
        self.step_number += 1
        action_scale = 0.01

        new_x, new_y, new_z = self.agent.get_position() + action[:3] * action_scale
        new_or_x, new_or_y, new_or_z = (self.agent.get_orientation() + action[3:]) % (2 * np.pi)
        # print("change in orientation: ", action[3:])
        # print("new orientation: ", new_orientation)
        # print("goal orientation: ", self.goal_orientation)


        # If within range, move
        if new_x > -0.25 and new_x < 0.25 and new_y > -0.25 and new_y < 0.25 and new_z > 0.8 and new_z < 2:
            self.agent.set_position([new_x, new_y, new_z])

        min_orientation = -np.pi/2
        max_orientation = np.pi/2

        if new_or_x > min_orientation and new_or_x < max_orientation and new_or_y > min_orientation and new_or_y < max_orientation and new_or_z > min_orientation and new_or_z < max_orientation:
            self.agent.set_orientation([new_or_x, new_or_y, new_or_z])

        distance = self.get_distance_to_goal()
        orientation_difference = self.get_orientation_difference_to_goal()
        # print(distance, orientation_difference)
        reward = - (distance + orientation_difference)

        done = False
        truncated = False

        if distance < 0.01:
            print("Reached goal distance!")
            reward = 50
        if self.get_orientation_diff_x() < 0.1 and self.get_orientation_diff_y() < 0.1 and self.get_orientation_diff_z() < 0.1:
            print("Reached goal orientation!")
            reward = 50
        if distance < 0.01 and self.get_orientation_diff_x() < 0.1 and self.get_orientation_diff_y() < 0.1 and self.get_orientation_diff_z() < 0.1:
            print("Reached goal!!")
            done = True
            reward = 200
        if self.step_number == 500:
            print("Failed to reach goal")
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

        # self.goal_pos = self.get_random_goal_pos()
        # self.goal_orientation = self.get_random_goal_orientation()

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())
        self.target.set_position(self.initial_target_pos)

        # self.goal_image = self._get_current_image(self.goal_camera)
        # img = Image.fromarray(self.goal_image)
        # img.save("goal_image" + str(self.goal_pos) + ".jpg")

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
        # print("agent pos:", [x, y, z])
        return [x, y, z]

    def get_random_agent_orientation(self):
        x = np.random.uniform(-np.pi, np.pi)
        y = np.random.uniform(-np.pi, np.pi)
        z = np.random.uniform(-np.pi, np.pi)
        # print("agent orientation:", [x, y, z])
        return [x, y, z]
    
    def get_distance_to_goal(self):
        x, y, z = self.agent.get_position()
        tx, ty, tz = self.goal_pos
        return np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)

    def get_orientation_difference_to_goal(self):
        return (self.get_orientation_diff_x() + self.get_orientation_diff_y() + self.get_orientation_diff_z())/(6*np.pi)

    def get_orientation_diff_x(self):
        x, _, _ = self.agent.get_orientation()
        tx, _, _ = self.goal_orientation
        return abs(x - tx)

    def get_orientation_diff_y(self):
        _, y, _ = self.agent.get_orientation()
        _, ty, _ = self.goal_orientation
        return abs(y - ty)

    def get_orientation_diff_z(self):
        _, _, z = self.agent.get_orientation()
        _, _, tz = self.goal_orientation
        return abs(z - tz)