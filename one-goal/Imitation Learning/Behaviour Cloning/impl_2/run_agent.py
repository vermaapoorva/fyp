from behavioural_cloning import train_model, run_model, get_number_of_parameters, create_model
from expert import collect_data
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import robot_env
import numpy as np
from keras.models import load_model
import os

if __name__ == "__main__":

    logdir = "/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/expert_data/"

    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
                ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
                ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
                ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
                ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
                ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

    scene_index = 3
    
    env = Monitor(gym.make("RobotEnv-v2", file_name=scenes[scene_index][0], bottleneck=scenes[scene_index][1]), "logs/")

    best_coeffs = None
    best_mean_return = -100000000
    best_hyperparams = None
    # Pos and or coeffs
    pos_and_or_coeffs = [[0.01, 1], [0.05, 1], [0.1, 1], [0.2, 1], [0.3, 1], [0.4, 1], [0.5, 1], [0.6, 1], [0.7, 1], [0.8, 1], [0.9, 1], [1, 1], [1, 0.01], [1, 0.05], [1, 0.1], [1, 0.2], [1, 0.3], [1, 0.4], [1, 0.5], [1, 0.6], [1, 0.7], [1, 0.8], [1, 0.9], [1, 0.99]]

    # Train model for hyperparameter and scene
    for hp, coeffs in enumerate(pos_and_or_coeffs):
    # for hp in range(1, 7):
        name_of_run = f"_10_train_loss_pos_{coeffs[0]}_or_{coeffs[1]}"

        # Get random hyperparameters
        batch_size = np.random.choice([1, 2, 4, 8])
        net_arch_layers = np.random.choice([3, 4])
        net_arch = []
        for j in range(net_arch_layers):
            net_arch.append(np.random.choice([32, 48, 64, 96, 128]))
        dense_arch = []
        dense_arch_layers = np.random.choice([2, 3, 4, 5])
        for k in range(dense_arch_layers):
            dense_arch.append(np.random.choice([100, 150, 200]))
        # learning_rate = np.random.choice(np.linspace(0.01, 0.1, 1000))
        learning_rate = np.random.choice(np.linspace(1e-8, 1e-2, 100))
        dropout_rate = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        # learning_rate = 1e-5
        learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        
        batch_size = 1
        net_arch = [32]
        dense_arch = [200]
        learning_rate = 1e-4
        dropout_rate = 0.2

        # Save hyperparameters to file in /vol/bitbucket/av1019/behavioural-cloning/hyperparameters/random_search
        # Create folder if doesn't exist
        # if not os.path.exists("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/random_search"):
        #     os.makedirs("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/random_search")
        # with open(f"/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/random_search/hyperparameters_{hp}.txt", "w") as f:
        #     f.write(f"batch_size: {batch_size}\n")
        #     f.write(f"net_arch: {net_arch}\n")
        #     f.write(f"dense_arch: {dense_arch}\n")
        #     f.write(f"learning_rate: {learning_rate}\n")
        #     f.write(f"dropout_rate: {dropout_rate}\n")

        scene_name = scenes[scene_index][0][:-4]

        print(f"Training model with hyperparameters: net_arch: {net_arch}, dropout_rate: {dropout_rate}, dense_arch: {dense_arch}, learning_rate: {learning_rate}, batch_size: {batch_size}")
        # print(f"Training model with coeffs: {coeffs}")

        mse = train_model(f"{name_of_run}_{hp}", # model_index
                    num_epochs = 2000, # num_epochs
                    pos_coeff = 0.3, # pos_coeff
                    or_coeff = 1, # or_coeff
                    batch_size = batch_size, # batch_size
                    dropout_rate = dropout_rate, # dropout_rate
                    learning_rate = learning_rate, # learning_rate
                    net_arch = net_arch, # net_arch
                    dense_arch = dense_arch, # dense_arch
                    amount_of_data = 500,
                    data_file = logdir + f"{scene_name}_expert_data_100000.pkl", # data_file
                    val_file = logdir + f"{scene_name}_expert_data_20000.pkl", # val_file
                    test_file = logdir + f"{scene_name}_expert_data_10000.pkl") # test_file

        # If model exists at path f"/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/models/net_arch_{hyperparameter_index}_object_{scene_index}_model.h5"
        # then run model
        # if os.path.exists(f"/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/models/net_arch_{hyperparameter_index}_object_{scene_index}_model.h5"):            
        mean_returns, mean_dist, mean_or =run_model(f"{name_of_run}_{hp}", env, num_of_runs=3)

        if mean_returns > best_mean_return:
            
            best_mean_return = mean_returns

            best_hyperparams = {"net_arch": net_arch,
                                "dense_arch": dense_arch,
                                "dropout_rate": dropout_rate,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "pos_coeff": 0.3,
                                "or_coeff": 1}
            print(f"New best mean return: {best_mean_return}")
        
        print(f"Best mean return: {best_mean_return}")
        print(f"Best hyperparameters: {best_hyperparams}")
