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
                ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]]]

        final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 64}

        scene_num = 2

        scene_name = scenes[scene_num][0]
        scene_bottleneck = scenes[scene_num][1]

        print(f"Training on scene: {scene_name}, bottleneck: {scene_bottleneck}")

        # file name without .ttt
        scene_file_name = scene_name[:-4]

        name_of_task = f"final_model_100k_{scene_file_name}"

        print(f"Training on task: {name_of_task}")

        trainer = ImageToPoseTrainerCoarse(task_name=name_of_task,
                                        hyperparameters=final_hyperparameters,
                                        scene_name=scene_file_name)

        start = time.process_time()
        trainer.train()
        end = time.process_time()

        average_steps, distance_error, orientation_error = run_model(task_name=name_of_task,
                                                        scene_name=scene_name,
                                                        bottleneck=scene_bottleneck,
                                                        hyperparameters=final_hyperparameters,
                                                        num_of_runs=50)

        results = []
        results.append({"scene_name": scene_file_name,
                        "seed": seed,
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
