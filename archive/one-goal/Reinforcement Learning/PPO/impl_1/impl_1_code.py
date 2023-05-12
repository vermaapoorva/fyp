import gym
from gym import spaces
from stable_baselines3 import PPO

import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Camera
	
SCENE_FILE = join(dirname(abspath(__file__)), 'impl_1_scene.ttt')

# class RobotEnv1(object):
class RobotEnv1(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(RobotEnv1, self).__init__()
        print("init")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(6)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-100, high=100,
                                            shape=(6,), dtype=float)

        self.done = False
        self.pr = PyRep()
        # Launch the application with a scene file in headless mode
        self.pr.launch(SCENE_FILE, headless=False) 
        self.pr.start()  # Start the simulation

        self.agent = VisionSensor("Camera")
        self.target = Object("Target")
        self.initial_agent_pos = self.agent.get_position()
        self.initial_target_pos = self.target.get_position()

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.concatenate([self.agent.get_position(),
                               self.target.get_position()])

    def step(self, action):

        self.pr.step()

        new_x, new_y, new_z = self.agent.get_position()
        if action == 0:
            new_x += 1
        elif action == 1:
            new_x -= 1
        elif action == 2:
            new_y += 1
        elif action == 3:
            new_y -= 1
        elif action == 4:
            new_z += 1
        elif action == 5:
            new_z -= 1
        
        self.agent.set_position([new_x, new_y, new_z])

        tx, ty, tz = self.target.get_position()
        reward = -np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2 + (new_z - tz) ** 2)

        done = False
        if reward > -0.5:
            done = True
            reward = 500

        return self._get_state(), reward, done, {}

    def reset(self):
        state = self._get_state()
        self.done = False
        self.agent.set_position(self.initial_agent_pos)
        self.target.set_position(self.initial_target_pos)
        # t = self.initial_target_pos
        # a = self.initial_agent_pos
        # reward = -np.sqrt((a[0] - t[0]) ** 2 + (a[1] - t[1]) ** 2 + (a[2] - t[2]) ** 2)
        # print("intial distance: ", reward)
        return self._get_state()  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close (self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

env = RobotEnv1()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=30000)
# model.save("ppo_cartpole")

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

    # print(i)

env.close()