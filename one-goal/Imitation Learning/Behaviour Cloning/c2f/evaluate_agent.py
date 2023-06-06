from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
from evaluate import run_model
import time
import json
import torch
import numpy as np

seed = 20

torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":

        scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
                ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
                ["bowl_scene.ttt", [-0.074, -0.023, 0.7745, -2.915]],
                ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]],
                ["purple_block_scene.ttt", [-0.015, 0.012, 0.720, 1.71042]]]

        final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.0001, "batch_size": 128}
        # final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 64}

        name_of_task = f"final_model_5M_fix_val_og_lr_bs_2048_cutlery_block_scene"
        scene_num = 0

        scene_name = scenes[scene_num][0]
        scene_bottleneck = scenes[scene_num][1]

        # file name without .ttt
        scene_file_name = scene_name[:-4]

        average_steps, distance_error, orientation_error, distance_std, or_std = run_model(task_name=name_of_task,
                                                                                        scene_name=scene_name,
                                                                                        bottleneck=scene_bottleneck,
                                                                                        hyperparameters=final_hyperparameters,
                                                                                        num_of_runs=50)

        results_file = f"/vol/bitbucket/av1019/behavioural-cloning/c2f/{name_of_task}.json"
        
        # read training time from results file if it exists
        if os.path.exists(results_file):
                with open(results_file, "r") as f:
                        training_time = json.load(f)[0].get("training_time", None)

        results = []
        results.append({"scene_name": scene_file_name,
                        "seed": seed,
                        "hyperparameters": final_hyperparameters,
                        "scene_bottleneck": scene_bottleneck,
                        "average_steps": average_steps,
                        "distance_error": distance_error,
                        "orientation_error": orientation_error,
                        "distance_std": distance_std,
                        "orientation_std": orientation_std,
                        "training_time": training_time})

        # Sort results by orientation error
        print(results)

        # Save results
        with open(results_file, "w") as f:
                json.dump(results, f, indent=4)
