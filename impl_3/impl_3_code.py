import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Camera
	
SCENE_FILE = join(dirname(abspath(__file__)), 'impl_3_scene.ttt')

#################################################################################################
################################   SETTING UP THE ENVIRONMENT   #################################
#################################################################################################

class RobotEnv3(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(RobotEnv3, self).__init__()
        print("init")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(6)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-100, high=100,
                                            shape=(256, 256, 3), dtype=float)

        self.done = False
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=True) 
        self.pr.start()  # Start the simulation

        self.agent = VisionSensor("Camera")
        self.target = Object("Target")
        self.initial_agent_pos = self.agent.get_position()
        self.initial_target_pos = self.target.get_position()
        self.step_number = 0

    def _get_state(self):
        # Return state containing image
        return self.normalise_image(self.agent.capture_rgb())

    def step(self, action):

        self.pr.step()
        self.step_number += 1

        new_x, new_y, new_z = self.agent.get_position()
        if action == 0:
            new_x += 0.01
        elif action == 1:
            new_x -= 0.01
        elif action == 2:
            new_y += 0.01
        elif action == 3:
            new_y -= 0.01
        elif action == 4:
            new_z += 0.01
        elif action == 5:
            new_z -= 0.01
        
        self.agent.set_position([new_x, new_y, new_z])

        tx, ty, tz = self.target.get_position()
        reward = -np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2 + (new_z - tz) ** 2)

        done = False
        if reward > -1:
            done = True
            reward = 500
        if self.step_number == 500:
            done = True
            self.step_number = 0

        return self._get_state(), reward, done, {}

    def reset(self):
        # state = self._get_state()
        self.done = False
        self.agent.set_position(self.initial_agent_pos)
        self.target.set_position(self.initial_target_pos)
        return self._get_state()  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close (self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

    def normalise_image(self, image):
        # divide each pixel by 255
        return image / 255

#################################################################################################
###################################   USING THE ENVIRONMENT   ###################################
#################################################################################################


env = RobotEnv3()

models_dir = "models/PPO_distance_0.01"
logdir = "logs/PPO_distance_0.01"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# # ### Training models with different timesteps
TIMESTEPS = 5000
for i in range(51):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/impl_3_{TIMESTEPS*i}")

results = []
for i in range(1, 51):
    model_path = f"{models_dir}/impl_3_{TIMESTEPS*i}.zip"
    model = PPO.load(model_path, env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
    results.append([TIMESTEPS*i, mean_reward, std_reward])

print(results)

model_path = f"{models_dir}/impl_3_100000.zip"
model = PPO.load(model_path, env=env)

episodes = 1000

for ep in range(episodes):
    obs = env.reset()
    done = False
    i = 0
    
    while not done and i <= 500:
        # pass observation to model to get predicted action
        action, _states = model.predict(obs)

        # pass action to env and get info back
        obs, rewards, done, info = env.step(action)

        # show the environment on the screen
        env.render()
        i += 1

    print(i)

env.close()