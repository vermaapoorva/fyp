from behavioural_cloning import train_model, run_model
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import robot_env

if __name__ == "__main__":
    position_orientation_coefficients = [[1, 0.01],
                                            [1, 0.1],
                                            [1, 0.2],
                                            [1, 0.3],
                                            [1, 0.4],
                                            [1, 0.5],
                                            [1, 0.6],
                                            [1, 0.7],
                                            [1, 0.8],
                                            [1, 0.9],
                                            [1, 1.0],
                                            [0.01, 1],
                                            [0.1, 1],
                                            [0.2, 1],
                                            [0.3, 1],
                                            [0.4, 1],
                                            [0.5, 1],
                                            [0.6, 1],
                                            [0.7, 1],
                                            [0.8, 1],
                                            [0.9, 1],
                                            [1.0, 1]]
    
    env = Monitor(gym.make("RobotEnv-v2"), "logs/")

    for position_orientation_coefficient in position_orientation_coefficients:
        print(f"Training model with position coefficient {position_orientation_coefficient[0]} and orientation coefficient {position_orientation_coefficient[1]}")
        train_model(f"model50_{position_orientation_coefficient[0]}_{position_orientation_coefficient[1]}", # model_index
                    position_orientation_coefficient[0], # position_coefficient
                    position_orientation_coefficient[1], # orientation_coefficient
                    100, # num_epochs
                    32, # batch_size
                    0.2, # dropout_rate
                    0.001, # learning_rate
                    "expert_data_50.pkl") # data_file

        run_model(f"model50_{position_orientation_coefficient[0]}_{position_orientation_coefficient[1]}", env)