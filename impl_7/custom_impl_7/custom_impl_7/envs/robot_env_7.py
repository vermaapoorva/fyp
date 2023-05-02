import gymnasium as gym
from gymnasium import spaces
import cv2

import numpy as np

from PIL import Image
from pyrep import PyRep
from pyrep.objects import VisionSensor, Object
from os.path import dirname, join, abspath

import time
SCENE_FILE = join(dirname(abspath(__file__)), 'impl_7_scene.ttt')

class RobotEnv7(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, headless=True, image_size=64, sleep=0):
        super(RobotEnv7, self).__init__()
        print("init")
        self.image_size = image_size
        self.sleep = sleep
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size*2), dtype=np.uint8)
        
        self.observation_space = spaces.Dict({"camera_image": spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8),
                                              "goal_image": spaces.Box(low=0, high=255,
                                            shape=(3, self.image_size, self.image_size), dtype=np.uint8)})

        self.done = False
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation

        self.agent = VisionSensor("camera")
        self.agent.set_explicit_handling(value=1)
        self.agent.handle_explicitly()
        self.target = Object("target")
        self.initial_agent_pos = self.agent.get_position()
        self.initial_target_pos = self.target.get_position()
        self.goal_pos = self.get_random_goal_pos()
        self.step_number = 0
        
        self.goal_image = self._get_goal_image()

        self.agent.set_position(self.get_random_agent_pos())

    def _get_goal_image(self):
        self.agent.set_position(self.goal_pos)
        goal_image = self._get_current_image()
        # img = Image.fromarray(goal_image.transpose(1, 2, 0))
        # img.save("goal_image_" + str(self.goal_pos) + ".jpg")
        return goal_image

    def _get_state(self):
        state = dict(camera_image=self._get_current_image(),
                     goal_image=self.goal_image)
        return state
        # state = np.concatenate((self._get_current_image(), self.goal_image), axis=1)
        # return state.transpose(2, 0, 1)

    def _get_current_image(self):
        image = self.agent.capture_rgb()
        resized = cv2.normalize(image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        resized = cv2.resize(resized, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        resized = resized.astype(np.uint8)
        return resized.transpose(2, 0, 1)

    def step(self, action):

        self.pr.step()
        self.step_number += 1
        action_scale = 0.01

        new_x, new_y, new_z = self.agent.get_position() + action_scale*action

        # If within range, move
        if new_x > -0.25 and new_x < 0.25 and new_y > -0.25 and new_y < 0.25 and new_z > 0.8 and new_z < 2:
            self.agent.set_position([new_x, new_y, new_z])

        tx, ty, tz = self.goal_pos
        reward = -np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2 + (new_z - tz) ** 2)

        done = False
        info = {}
        truncated = False
        if reward > -0.01:
            done = True
            reward = 200
            info.update({"success": True})
        if self.step_number == 200:
            done = True
            truncated = True
            self.step_number = 0

        # time.sleep(self.sleep)
        # if done:
        #     time.sleep(self.sleep * 100)
        
        self.target.set_position(self.initial_target_pos)

        return self._get_state(), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # state = self._get_state()
        self.done = False
        self.goal_pos = self.get_random_goal_pos()
        self.goal_image = self._get_goal_image()
        self.agent.set_position(self.get_random_agent_pos())
        self.target.set_position(self.initial_target_pos)
        # print("goal pos:", self.goal_pos)
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
    
    def get_random_goal_pos(self):
        x = np.random.uniform(-0.2, 0.2)
        y = np.random.uniform(-0.2, 0.2)
        z = np.random.uniform(1, 1.5)
        return [x, y, z]
    
    def get_distance_to_goal(self):
        x, y, z = self.agent.get_position()
        tx, ty, tz = self.goal_pos
        return np.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
