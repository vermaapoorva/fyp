from network import ImageToPoseNetworkCoarse
from dagger_agent_npy_mp import train_model
from evaluate import run_model

import time
import json
import torch
import numpy as np

import os

torch.manual_seed(20)
np.random.seed(20)

if __name__ == "__main__":

    scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
        ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
        ["bowl_scene.ttt", [-0.074, -0.023, 0.7745, -2.915]],
        ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]],
        ["purple_block_scene.ttt", [-0.015, 0.012, 0.720, 1.71042]]]

    final_hyperparameters = {"net_arch": [32, 64, 128, 256],
                                "learning_rate": 0.0001,
                                "batch_size": 128,
                                "amount_of_data": 1000000,
                                "num_dagger_iterations": 10}

    # final_hyperparameters = {"net_arch": [32, 48, 64, 128],
    #                             "learning_rate": 0.001,
    #                             "batch_size": 64,
    #                             "amount_of_data": 10000,
    #                             "num_dagger_iterations": 10}

    for scene_index in [3]:

        results = []

        scene_file_name = scenes[scene_index][0]
        scene_name = scene_file_name.split(".")[0]

        name_of_task = f"final_model_1M_10_iters_new_bc_params_fix_val_{scene_name}"

        # training_time = 0

        # start = time.process_time()
        train_model(task_name=name_of_task,
                    scene_file_name=scene_file_name,
                    bottleneck=scenes[scene_index][1],
                    hyperparameters=final_hyperparameters,
                    start_iteration=7,
                    training_shards_next_index=70,
                    validation_shards_next_index=14)
        # end = time.process_time()

        # print("Training time: " + str(end - start) + " seconds")

        distance_errors, orientation_errors = run_model(task_name=name_of_task,
                                                        scene_name=scenes[scene_index][0],
                                                        bottleneck=scenes[scene_index][1],
                                                        hyperparameters=final_hyperparameters,
                                                        num_of_runs=50)

        results_file = f"/vol/bitbucket/av1019/dagger/hyperparameters/{name_of_task}.json"

        # training_time = end - start
        # if training_time != 0 and os.path.exists(results_file):
        #     with open(results_file, "r") as f:
        #             training_time = json.load(f)[0].get("training_time", None)

        results.append({"scene_index": scene_index,
                        "net_arch": final_hyperparameters['net_arch'],
                        "learning_rate": final_hyperparameters['learning_rate'],
                        "batch_size": final_hyperparameters['batch_size'],
                        "distance_errors": distance_errors,
                        "orientation_errors": orientation_errors})

        # Sort results by orientation error
        print(results)

        # Save results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)