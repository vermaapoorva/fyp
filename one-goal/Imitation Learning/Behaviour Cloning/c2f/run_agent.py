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

        scene_num = 3

        scene_name = scenes[scene_num][0]
        scene_bottleneck = scenes[scene_num][1]

        final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 4096}

        print(f"Training on scene: {scene_name}, bottleneck: {scene_bottleneck}")
        start = 0
        end = 0

        hyperparameters = [{"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 256},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 512},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 1024},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 2048},
                        {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 4096}]

        amount_of_data = 5000000

        bs = final_hyperparameters["batch_size"]
        name_of_task = f"final_model_5M_fix_val_og_lr_bs_{bs}_{scene_file_name}"
        epoch = 5
        checkpoint_path = f"/vol/bitbucket/av1019/behavioural-cloning/c2f/Networks/final_model_5M_fix_val_og_lr_bs_4096_{scene_file_name}/network_checkpoint_epoch_{epoch}.torch"

                print(f"Training on scene: {scene_name}, bottleneck: {scene_bottleneck}")
                print(f"hyperparameters: {hyperparameter}")

        trainer = ImageToPoseTrainerCoarse(task_name=name_of_task,
                                        hyperparameters=final_hyperparameters,
                                        scene_name=scene_file_name,
                                        checkpoint_path=checkpoint_path,
                                        starting_epoch=epoch)

                # file name without .ttt
                scene_file_name = scene_name[:-4]

        distance_errors, orientation_errors = run_model(task_name=name_of_task,
                                                        scene_name=scene_name,
                                                        bottleneck=scene_bottleneck,
                                                        hyperparameters=final_hyperparameters,
                                                        num_of_runs=50)

        results = []
        results.append({"scene_name": scene_file_name,
                        "seed": seed,
                        "amount_of_data": amount_of_data,
                        "hyperparameters": final_hyperparameters,
                        "scene_bottleneck": scene_bottleneck,
                        "average_steps": average_steps,
                        "distance_errors": distance_errors,
                        "orientation_errors": orientation_errors,
                        "training_time": end - start})

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
                                "hyperparameters": final_hyperparameters,
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
