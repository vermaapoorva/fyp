import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np
from PIL import Image

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Shape
from os.path import dirname, join, abspath

import time
SCENE_FILE = join(dirname(abspath(__file__)), 'robot_env.ttt')

# Bottleneck position+orientation
BOTTLENECK_X = -0.06
BOTTLENECK_Y = 0.04
BOTTLENECK_Z = 0.7
BOTTLENECK_ORIENTATION_Z = -np.pi/9

# Range of the agent's position
MAX_HEIGHT = 1
MAX_RADIUS = 0.2
MIN_RADIUS = 0.1

# Max action length for 1 step
MAX_ACTION = 0.01

# Max episode length (in steps)
MAX_EPISODE_LENGTH = 500

# Thresholds
DISTANCE_THRESHOLD = 0.001
ANGLE_THRESHOLD = 0.01

class RobotEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self,
                headless=True,
                image_size=64,
                sleep=0,
                file_name="robot_env.ttt",
                bottleneck=[BOTTLENECK_X, BOTTLENECK_Y, BOTTLENECK_Z, BOTTLENECK_ORIENTATION_Z]):
        super(RobotEnv, self).__init__()
        print("init")

        SCENE_FILE = join(dirname(abspath(__file__)), "scenes/" + file_name)
        BOTTLENECK_X, BOTTLENECK_Y, BOTTLENECK_Z, BOTTLENECK_ORIENTATION_Z = bottleneck

        self.image_size = image_size
        self.sleep = sleep
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -np.pi/4]),
                                       high=np.array([1, 1, 1, np.pi/4]),
                                       shape=(4,),
                                       dtype=np.float64)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8)
        
        # Launch the application with a scene file in headless mode
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation
        
        self.step_number = 0
        self.done = False

        self.table = Shape("table")
        self.agent = VisionSensor("camera")
        self.goal_camera = VisionSensor("goal_camera")
        self.target = Object("target")
        self.initial_target_pos = self.target.get_position()

        # Set goal position+orientation and capture goal image
        self.goal_pos = [BOTTLENECK_X, BOTTLENECK_Y, BOTTLENECK_Z]
        self.goal_orientation = [-np.pi, 0, BOTTLENECK_ORIENTATION_Z]
        self.goal_camera.set_position(self.goal_pos)
        self.goal_camera.set_orientation(self.goal_orientation)
        img = Image.fromarray(self._get_current_image(self.goal_camera))
        img.save("goal.jpg")

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())

        self.max_distance_to_goal = self.get_max_distance_to_goal()

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

        # Normalize action to MAX_ACTION
        pos_action = action[:3]
        length = np.linalg.norm(pos_action)
        if length > MAX_ACTION:
            pos_action *= MAX_ACTION/length
        
        # Calculate new position and orientation
        new_x, new_y, new_z = self.agent.get_position() + pos_action
        curr_or_x, curr_or_y, curr_or_z = self.agent.get_orientation()
        new_or_z = (curr_or_z + action[3]) % (2 * np.pi)

        # If within range, move agent
        radius = self.get_current_radius()
        if abs(new_x) > radius:
            new_x = radius * np.sign(new_x)
        if abs(new_y) > radius:
            new_y = radius * np.sign(new_y)
        if new_z < self.goal_pos[2]:
            new_z = self.goal_pos[2]
        if new_z > MAX_HEIGHT:
            new_z = MAX_HEIGHT
        self.agent.set_position([new_x, new_y, new_z])

        # Rotate agent
        self.agent.set_orientation([curr_or_x, curr_or_y, new_or_z])

        # Calculate reward
        dist_factor = 0.9
        or_factor = 0.1

        distance = self.get_distance_to_goal()/self.max_distance_to_goal
        orientation_difference = self.get_orientation_diff_z()/np.pi
        reward = - (distance*dist_factor + orientation_difference*or_factor)

        done = False
        truncated = False

        # if self.get_distance_to_goal() < DISTANCE_THRESHOLD:
        #     print("Reached goal distance!")
            # reward = 10
        # if self.get_orientation_diff_z() < ANGLE_THRESHOLD:
        #     print("Reached goal orientation!")
            # reward = 1
        if self.get_distance_to_goal() < DISTANCE_THRESHOLD and self.get_orientation_diff_z() < ANGLE_THRESHOLD:
            # print("Reached goal!!")
            done = True
            reward = 200
        if self.step_number == MAX_EPISODE_LENGTH:
            done = True
            truncated = True
            self.step_number = 0

        time.sleep(self.sleep)
        if done:
            time.sleep(self.sleep * 100)

        return self._get_state(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_number = 0
        self.done = False

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())
        
        self.target.set_position(self.initial_target_pos)

        return self._get_state(), {}

    def render(self, mode='human'):
        return

    def close (self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

    def get_random_agent_pos(self):
        radius = self.get_current_radius(initial=True)
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        z = MAX_HEIGHT
        return [x, y, z]

    def get_random_agent_orientation(self):
        x = self.goal_orientation[0]
        y = self.goal_orientation[1]
        z = np.random.uniform(-np.pi, np.pi)
        # print("agent orientation:", [x, y, z])
        return [x, y, z]
    
    def get_distance_to_goal(self):
        x, y, z = self.agent.get_position()
        tx, ty, tz = self.goal_pos
        return np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)

    # Returns smallest absolute difference between current and goal orientation
    def get_orientation_diff_z(self):
        _, _, z = self.agent.get_orientation()
        _, _, tz = self.goal_orientation
        diff = abs(z - tz)
        return min(diff, 2 * np.pi - diff)

    def get_current_radius(self, initial=False):
        min_height = self.goal_pos[2]
        current_height = self.agent.get_position()[2]

        if initial:
            current_height = MAX_HEIGHT

        return MIN_RADIUS + ((current_height - min_height) / (MAX_HEIGHT - min_height)) * (MAX_RADIUS - MIN_RADIUS)

    def get_max_distance_to_goal(self):
        max_radius = self.get_current_radius(initial=True)

        corner_1 = np.linalg.norm(np.array([max_radius, max_radius, MAX_HEIGHT]) - np.array(self.goal_pos))
        corner_2 = np.linalg.norm(np.array([-max_radius, max_radius, MAX_HEIGHT]) - np.array(self.goal_pos))
        corner_3 = np.linalg.norm(np.array([max_radius, -max_radius, MAX_HEIGHT]) - np.array(self.goal_pos))
        corner_4 = np.linalg.norm(np.array([-max_radius, -max_radius, MAX_HEIGHT]) - np.array(self.goal_pos))

        max_distance_to_goal = max(corner_1, corner_2, corner_3, corner_4)
        return max_distance_to_goal
