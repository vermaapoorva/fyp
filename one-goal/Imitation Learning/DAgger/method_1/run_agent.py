from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
from dagger_agent import train_model
from evaluate import run_model

import time
import json
import torch
import numpy as np

torch.manual_seed(20)
np.random.seed(20)

scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
            ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
            ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
            ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
            ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
            ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

hyperparameters = [ {"net_arch": [64, 128, 256], "learning_rate": 0.001, "batch_size": 32},
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
                    {"net_arch": [64, 128, 128, 256], "learning_rate": 0.00001, "batch_size": 64}]

# Network architecture: [64, 128, 256], Number of trainable parameters: 523920
# Network architecture: [32, 64, 128, 256], Number of trainable parameters: 541520
# Network architecture: [64, 128, 128, 256], Number of trainable parameters: 671504

original_hyperparameters = {"net_arch": [32, 48, 64, 128],
                            "learning_rate": 0.001,
                            "batch_size": 32,
                            "num_rollouts": 5,
                            "num_dagger_iterations": 20}

# scene_index = 0

results = []

# for i, hyperparameter in enumerate(hyperparameters):
for scene_index, scene in enumerate(scenes):
    hyperparameter = original_hyperparameters

    # task_name = f"tuning_hp_og_patiences_results_scene_{scene_index}_hp_{i}"
    task_name = f"original_hp_results_scene_{scene_index}"

    scene_file_name, bottleneck = scenes[scene_index]

    start = time.process_time()
    train_model(task_name, scene_file_name, bottleneck, hyperparameter)
    end = time.process_time()

    distance_error, orientation_error = run_model(task_name=task_name,
                                                    scene_name=scene_file_name,
                                                    bottleneck=bottleneck,
                                                    hyperparameters=hyperparameter,
                                                    num_of_runs=20)

    results.append({"net_arch": hyperparameter['net_arch'],
                    "learning_rate": hyperparameter['learning_rate'],
                    "batch_size": hyperparameter['batch_size'],
                    "distance_error": distance_error,
                    "orientation_error": orientation_error,
                    "training_time": end - start})

    # Sort results by orientation error
    results = sorted(results, key=lambda k: k['orientation_error'])
    print(results)

    # Save results
    with open(f"/vol/bitbucket/av1019/dagger/hyperparameters/original_hp_results_all_scenes.json", "w") as f:
        json.dump(results, f, indent=4)