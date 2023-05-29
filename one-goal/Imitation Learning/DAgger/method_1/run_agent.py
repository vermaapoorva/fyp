from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
# from dagger_agent_npy import train_model
from dagger_agent_npy_mp import train_model
from evaluate import run_model

import time
import json
import torch
import numpy as np

torch.manual_seed(20)
np.random.seed(20)

if __name__ == "__main__":

    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
                ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
                ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
                ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
                ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
                ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

    hyperparameters = [ {"net_arch": [32, 48, 64, 128], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [32, 48, 64, 128], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [32, 48, 64, 128], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [32, 48, 64, 128], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [32, 48, 64, 128], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [32, 48, 64, 128], "learning_rate": 0.00001, "batch_size": 64},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [64, 128, 256], "learning_rate": 0.00001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.00001, "batch_size": 64},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [64, 128, 128, 256], "learning_rate": 0.00001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [32, 64, 128, 64], "learning_rate": 0.00001, "batch_size": 64},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.001, "batch_size": 32},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.0001, "batch_size": 32},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.00001, "batch_size": 32},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.001, "batch_size": 64},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.0001, "batch_size": 64},
                        {"net_arch": [64, 128, 64], "learning_rate": 0.00001, "batch_size": 64}]

    final_hyperparameters = {"net_arch": [32, 64, 128, 256],
                                "learning_rate": 0.001,
                                "batch_size": 64,
                                "amount_of_data": 10000,
                                "num_dagger_iterations": 10}

    # scene_index = 0

    for i in range(24, 36):

        results = []

        hyperparameter = hyperparameters[i]
        hyperparameter["num_dagger_iterations"] = 10
        hyperparameter["amount_of_data"] = 10000

        hyperparameter = final_hyperparameters

        for scene_index, scene in enumerate(scenes):

            # name_of_task = f"10000_tuning_scene_{scene_index}_hp_{i}"

            scene_file_name = scenes[scene_index][0]
            scene_name = scene_file_name.split(".")[0]

            # name_of_task = f"try_webdataset_10k_10_val_1_env_10_iters"

            name_of_task = f"final_tuning_scene_{scene_index}_hp_{i}"

            start = time.process_time()
            train_model(task_name=name_of_task,
                        scene_file_name=scene_file_name,
                        bottleneck=scene[1],
                        hyperparameters=hyperparameter)
            end = time.process_time()

            print("Training time: " + str(end - start) + " seconds")

            average_steps, distance_error, orientation_error = run_model(task_name=name_of_task,
                                                            scene_name=scenes[scene_index][0],
                                                            bottleneck=scenes[scene_index][1],
                                                            hyperparameters=hyperparameter,
                                                            num_of_runs=20)

            results.append({"scene_index": scene_index,
                            "net_arch": hyperparameter['net_arch'],
                            "learning_rate": hyperparameter['learning_rate'],
                            "batch_size": hyperparameter['batch_size'],
                            "distance_error": distance_error,
                            "orientation_error": orientation_error,
                            "training_time": end - start})

            # Sort results by orientation error
            print(results)

            # Save results
            with open(f"/vol/bitbucket/av1019/dagger/hyperparameters/final_tuning_hp_{i}.json", "w") as f:
                json.dump(results, f, indent=4)