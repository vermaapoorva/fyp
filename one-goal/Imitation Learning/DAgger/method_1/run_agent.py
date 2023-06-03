from network import ImageToPoseNetworkCoarse
from dagger_agent_npy_mp import train_model
from evaluate import run_model

import time
import json
import torch
import numpy as np

torch.manual_seed(20)
np.random.seed(20)

if __name__ == "__main__":

    scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
        ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
        ["bowl_scene.ttt", [-0.074, -0.023, 0.7745, -2.915]],
        ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]]]

    final_hyperparameters = {"net_arch": [32, 48, 64, 128],
                                "learning_rate": 0.001,
                                "batch_size": 32,
                                "amount_of_data": 10000000,
                                "num_dagger_iterations": 20}

    scene_index = 3

    results = []

    scene_file_name = scenes[scene_index][0]
    scene_name = scene_file_name.split(".")[0]

    name_of_task = f"final_model_10M_20_iters_{scene_name}"

    start = time.process_time()
    train_model(task_name=name_of_task,
                scene_file_name=scene_file_name,
                bottleneck=scenes[scene_index][1],
                hyperparameters=final_hyperparameters,
                start_iteration=0,
                training_shards_next_index=0,
                validation_shards_next_index=0)
    end = time.process_time()

    print("Training time: " + str(end - start) + " seconds")

    average_steps, distance_error, orientation_error = run_model(task_name=name_of_task,
                                                    scene_name=scenes[scene_index][0],
                                                    bottleneck=scenes[scene_index][1],
                                                    hyperparameters=final_hyperparameters,
                                                    num_of_runs=50)

    results.append({"scene_index": scene_index,
                    "net_arch": final_hyperparameters['net_arch'],
                    "learning_rate": final_hyperparameters['learning_rate'],
                    "batch_size": final_hyperparameters['batch_size'],
                    "distance_error": distance_error,
                    "orientation_error": orientation_error,
                    "training_time": end - start})

    # Sort results by orientation error
    print(results)

    # Save results
    with open(f"/vol/bitbucket/av1019/dagger/hyperparameters/{name_of_task}.json", "w") as f:
        json.dump(results, f, indent=4)