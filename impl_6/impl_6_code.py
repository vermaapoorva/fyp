import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import cv2

import numpy as np
from os.path import dirname, join, abspath

from pyrep import PyRep
from pyrep.objects import VisionSensor, Object, Camera

import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

SCENE_FILE = join(dirname(abspath(__file__)), 'impl_6_scene.ttt')

#################################################################################################
################################   SETTING UP THE ENVIRONMENT   #################################
#################################################################################################

class RobotEnv6(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, headless=True):
        super(RobotEnv6, self).__init__()
        print("init")
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(36, 36, 3), dtype=np.uint8)

        self.done = False
        self.pr = PyRep()
        # Launch the application with a scene file i
        # n headless mode
        self.pr.launch(SCENE_FILE, headless=headless) 
        self.pr.start()  # Start the simulation

        self.agent = VisionSensor("Camera")
        self.target = Object("Target")
        self.initial_agent_pos = self.agent.get_position()
        self.initial_target_pos = self.target.get_position()
        self.goal_pos = [0, 0.5, 0.5]
        self.step_number = 0
        
        self.agent.set_position(self.get_random_agent_pos())


    def _get_state(self):
        # Return state containing image
        image = self.agent.capture_rgb()
        resized = cv2.resize(image, (36, 36), interpolation = cv2.INTER_AREA)
        return resized

    def step(self, action):

        self.pr.step()
        self.step_number += 1
        action_scale = 0.01

        new_x, new_y, new_z = self.agent.get_position()
        self.agent.set_position([new_x + action_scale*action[0], new_y + action_scale*action[1], new_z + action_scale*action[2]])

        tx, ty, tz = self.goal_pos
        reward = -np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2 + (new_z - tz) ** 2)

        done = False
        if new_x < -2.5 or new_x > 2.5 or new_y < -2.5 or new_y > 2.5 or new_z < 0 or new_z > 2.5:
            done = True
            reward = -5000
        if reward > -0.01:
            done = True
            reward = 500
        if self.step_number == 500:
            done = True
            self.step_number = 0
        
        return self._get_state(), reward, done, {}

    def reset(self):
        # state = self._get_state()
        self.done = False
        random_agent_pos = self.get_random_agent_pos()
        self.agent.set_position(random_agent_pos)
        self.target.set_position(self.initial_target_pos)
        return self._get_state()  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close (self):
        self.pr.stop()  # Stop the simulation
        self.pr.shutdown()  # Close the application

    def get_random_agent_pos(self):
        # full range to see object: x(-1.5, 1.5), y(1, 2.5), z(0, 1.3)
        # required range: x 0.5, y 0.5, z 0.5-1
        # These work, stored in PPO_1 + Average final reward_1
        # x = np.random.uniform(-0.25, 0.25)
        # y = np.random.uniform(1, 1.25)
        # z = np.random.uniform(0, 1)
        # Required range works, stored in PPO_2 + Average final reward_2
        x = np.random.uniform(-0.25, 0.25)
        y = np.random.uniform(1, 1.5)
        z = np.random.uniform(0.5, 1)
        return [x, y, z]

#################################################################################################
##########################################   CALLBACK   #########################################
#################################################################################################

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param model_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, logdir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.logdir = logdir
        self.save_path = os.path.join(logdir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.logdir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True
    
# Define a custom callback function to calculate the average reward per time step
class AvgRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.avg_reward = 0
        self.results = []

    def _on_step(self) -> bool:
        if self.locals.get("dones")[0]:
            result = 0
            if self.locals.get("rewards")[0] == 500:
                result = 1
            tensorboard_callback.writer.add_scalar('Final reward', result, self.num_timesteps)
            self.results.append(result)
        return True

    def _on_training_end(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        print("rollout ends after {} episodes".format(len(self.results)))
        tensorboard_callback.writer.add_scalar('Average final reward', np.mean(self.results), self.num_timesteps)
        self.results = []

#################################################################################################
###################################   USING THE ENVIRONMENT   ###################################
#################################################################################################


logdir = "logs"
tensorboard_log_dir = "tensorboard_logs"
tensorboard_callback = TensorBoardOutputFormat(tensorboard_log_dir + "/Average final reward_1")

def train():

    env = RobotEnv6()
    env = Monitor(env, logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, device='cuda')

    # Create the callbacks
    save_best_model_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, logdir=logdir)
    avg_reward_callback = AvgRewardCallback()

    # Train the agent
    timesteps = 50000000
    model.learn(total_timesteps=int(timesteps), callback=[save_best_model_callback, avg_reward_callback])
    plot_results([logdir], timesteps, results_plotter.X_TIMESTEPS, "PPO")
    plt.show()

def run_model():

    env = RobotEnv6(False)
    env = Monitor(env, logdir)

    model_path = f"{logdir}/best_model.zip"
    model = PPO.load(model_path, env=env)

    episodes = 1000

    for ep in range(episodes):
        obs = env.reset()
        done = False
        i = 0
        episode_rewards = []
        while not done and i <= 500:
            # pass observation to model to get predicted action
            action, _states = model.predict(obs)

            # pass action to env and get info back
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            # show the environment on the screen
            env.render()
            i += 1

        # print(i)
        print(np.mean(np.sum(episode_rewards)))

train()
# run_model()