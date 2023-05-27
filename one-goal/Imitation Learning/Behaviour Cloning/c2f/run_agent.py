from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
from evaluate import run_model
import time
import json
import torch
import numpy as np

torch.manual_seed(20)
np.random.seed(20)

scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
        ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]]]

# scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056], 0]]

final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 64}
for scene_name, scene_bottleneck in scenes:

    print(f"Training on scene: {scene_name}, bottleneck: {scene_bottleneck}")

    # file name without .ttt
    scene_file_name = scene_name[:-4]
    
    name_of_task = f"try_cutlery_block2_with_mp_npy_12000_{scene_file_name}"

    trainer = ImageToPoseTrainerCoarse(task_name=name_of_task,
                                        hyperparameters=final_hyperparameters,
                                        amount_of_data=12000,
                                        scene_name=scene_file_name)

    start = time.process_time()
    trainer.train()
    end = time.process_time()

    average_steps, distance_error, orientation_error = run_model(task_name=name_of_task,
                                                    scene_name=scene_name,
                                                    bottleneck=scene_bottleneck,
                                                    hyperparameters=final_hyperparameters,
                                                    num_of_runs=20)

    results.append({"scene_name": scene_file_name,
                    "scene_bottleneck": scene_bottleneck,
                    "average_steps": average_steps,
                    "distance_error": distance_error,
                    "orientation_error": orientation_error,
                    "training_time": end - start})

    # Sort results by orientation error
    print(results)

    # Save results
    with open(f"/vol/bitbucket/av1019/behavioural-cloning/c2f/{name_of_task}.json", "w") as f:
        json.dump(results, f, indent=4)