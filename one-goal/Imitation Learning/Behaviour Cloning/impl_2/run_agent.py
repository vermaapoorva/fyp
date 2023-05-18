from behavioural_cloning import train_model, run_model
from expert import collect_data
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import robot_env
import numpy as np

if __name__ == "__main__":

    hyperparameters = [{"num_epochs": 30, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.005},
                        {"num_epochs": 30, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
                        {"num_epochs": 30, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0005},
                        {"num_epochs": 30, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0001},
                        {"num_epochs": 40, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.005},
                        {"num_epochs": 40, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
                        {"num_epochs": 40, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0005},
                        {"num_epochs": 40, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0001},
                        {"num_epochs": 50, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.005},
                        {"num_epochs": 50, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
                        {"num_epochs": 50, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0005},
                        {"num_epochs": 50, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0001},
                        {"num_epochs": 75, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.005},
                        {"num_epochs": 75, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
                        {"num_epochs": 75, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0005},
                        {"num_epochs": 75, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0001},
                        {"num_epochs": 100, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.005},
                        {"num_epochs": 100, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
                        {"num_epochs": 100, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0005},
                        {"num_epochs": 100, "batch_size": 32, "dropout_rate": 0.2, "learning_rate": 0.0001}]

    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
                ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
                ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
                ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
                ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
                ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

    position_orientation_coefficients = [[1, 0.01], [1, 0.1], [1, 0.2], [1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6], [1, 0.7], [1, 0.8], [1, 0.9], [1, 1],
                                        [0.01, 1], [0.1, 1], [0.2, 1], [0.3, 1], [0.4, 1], [0.5, 1], [0.6, 1], [0.7, 1], [0.8, 1], [0.9, 1], [1, 1]]

    # env = Monitor(gym.make("RobotEnv-v2", file_name=scenes[0], bottleneck=scenes[1]), "logs/")

    scene_index = 2
    collect_data(scenes[scene_index][0], scenes[scene_index][1], 100000)

    # for position_orientation_coefficient in position_orientation_coefficients:
    #     print(f"Training model with position coefficient {position_orientation_coefficient[0]} and orientation coefficient {position_orientation_coefficient[1]}")
    #     train_model(f"model500_{position_orientation_coefficient[0]}_{position_orientation_coefficient[1]}", # model_index
    #                 30, # num_epochs
    #                 32, # batch_size
    #                 0.2, # dropout_rate
    #                 0.001, # learning_rate
    #                 "expert_data_500.pkl") # data_file

    #     run_model(f"model500_{position_orientation_coefficient[0]}_{position_orientation_coefficient[1]}", env)