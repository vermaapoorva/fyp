import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Shape
from pyrep.const import PrimitiveShape
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

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 file_name,
                 bottleneck,
                 headless=True,
                 image_size=64,
                 sleep=0,
                 evaluate=False):
        super(RobotEnv, self).__init__()
        self.metadata = {'render_modes': ['rgb_array']}
        print("initialising robot env...")

        SCENE_FILE = join(dirname(abspath(__file__)), "scenes/" + file_name)
        BOTTLENECK_X, BOTTLENECK_Y, BOTTLENECK_Z, BOTTLENECK_ORIENTATION_Z = bottleneck

        self.image_size = image_size
        self.sleep = sleep
        self.eval = evaluate
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -np.pi/4]),
                                       high=np.array([1, 1, 1, np.pi/4]),
                                       shape=(4,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8)
        
        # Launch the application with a scene file in headless mode
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()  # Start the simulation

        self.step_number = 0

        self.table = Shape("table")
        self.agent = VisionSensor("camera")
        self.goal_camera = VisionSensor("goal_camera")
        self.target = Shape("Shape")
        self.initial_target_pos = self.target.get_position()

        # Set goal position+orientation and capture goal image
        self.goal_pos = [BOTTLENECK_X, BOTTLENECK_Y, BOTTLENECK_Z]
        self.goal_orientation = [-np.pi, 0, BOTTLENECK_ORIENTATION_Z]
        self.goal_camera.set_position(self.goal_pos)
        self.goal_camera.set_orientation(self.goal_orientation)

        self.save_goal_image(file_name)
        self.save_initial_image(file_name)

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())

        self.max_distance_to_goal = self.get_max_distance_to_goal()

    def save_goal_image(self, file_name):
        self.agent.set_position(self.goal_pos)
        self.agent.set_orientation(self.goal_orientation)
        self.pr.step()
        image = self.agent.capture_rgb()
        # print("image:", image)
        scene_name_without_ttt = file_name[:-4]
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        plt.imsave("goal_" + scene_name_without_ttt + ".png", resized)

    def save_initial_image(self, file_name):
        self.agent.set_position([0, 0, 1])
        self.agent.set_orientation([-np.pi, 0, 0])
        self.pr.step()
        image = self.agent.capture_rgb()
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        scene_name_without_ttt = file_name[:-4]
        plt.imsave("intial_" + scene_name_without_ttt + ".png", resized)

    def set_goal(self, goal_pos, goal_orientation):
        self.goal_pos = goal_pos
        self.goal_orientation = goal_orientation

    def set_agent(self, agent_pos, agent_orientation):
        self.agent.set_position(agent_pos)
        self.agent.set_orientation(agent_orientation)

    def get_agent_position(self):
        return self.agent.get_position()
    
    def get_agent_orientation(self):
        return self.agent.get_orientation()

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
        if action is None:
            return self._get_state(), 0.0, False, False, {}

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

        if self.eval:
            return self._get_state(), reward, False, truncated, {}

        return self._get_state(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        final_distance_to_goal = self.get_distance_to_goal()
        final_orientation_diff_z = self.get_orientation_diff_z()
        self.step_number = 0

        self.agent.set_position(self.get_random_agent_pos())
        self.agent.set_orientation(self.get_random_agent_orientation())
        
        self.target.set_position(self.initial_target_pos)

        return self._get_state(), {"final_distance": final_distance_to_goal, "final_orientation": final_orientation_diff_z}

    def render(self, mode='rgb_array'):
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
        z = np.random.uniform(-np.pi/4, np.pi/4)
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
    
    def draw_distribution(self):
        self.agent.set_position([0, 0, MAX_HEIGHT])
        current_height = MAX_HEIGHT
        shape = 0.01
        while self.agent.get_position()[2] >= self.goal_pos[2]:
            current_radius = self.get_current_radius()
            print("Current radius = ", current_radius)
            x_range = np.arange(-current_radius, current_radius, shape)
            y_range = np.arange(-current_radius, current_radius, shape)
            if x_range[-1] < current_radius:
                x_range = np.append(x_range, current_radius)
            if y_range[-1] < current_radius:
                y_range =np.append(y_range, current_radius)
            for x in x_range:
                for y in y_range:
                    self.pr.step()
                    self.agent.set_position([x, y, current_height])
                    self.draw_shape(shape)
            
            current_height -= shape
            self.agent.set_position([0, 0, current_height])
            self.pr.step()           
                    
    def draw_shape(self, shape):
        current = Shape.create(PrimitiveShape.CUBOID,
                               size=[shape, shape, shape],
                               position=self.agent.get_position(),
                               color=[0, 255, 0])
        
        current.set_dynamic(False)
        current.set_renderable(False)
        current.set_collidable(False)
        current.set_detectable(False)


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
