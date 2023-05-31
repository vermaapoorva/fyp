from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
from evaluate import run_model
import time
import json
import torch
import numpy as np

torch.manual_seed(20)
np.random.seed(20)

# scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
#         ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
#         ["bowl_scene.ttt", [-0.074, -0.023, 0.7745, -2.915]],
#         ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]]]

scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
        ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
        ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
        ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
        ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
        ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

final_hyperparameters = {"net_arch": [32, 64, 128, 256], "learning_rate": 0.001, "batch_size": 64}
for scene_name, scene_bottleneck in scenes[0:1]:

    print(f"Training on scene: {scene_name}, bottleneck: {scene_bottleneck}")

    # file name without .ttt
    scene_file_name = scene_name[:-4]
    
    # amount_of_data = 10000

    name_of_task = f"test_10k_10k_og_params_tuning_seed_10cm_mp_1_data_{scene_file_name}"

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
                                                    num_of_runs=20)

    # results.append({"scene_name": scene_file_name,
    #                 "scene_bottleneck": scene_bottleneck,
    #                 "average_steps": average_steps,
    #                 "distance_error": distance_error,
    #                 "orientation_error": orientation_error,
    #                 "amount_of_data": amount_of_data,
    #                 "training_time": end - start})

    # # Sort results by orientation error
    # print(results)

    # # Save results
    # with open(f"/vol/bitbucket/av1019/behavioural-cloning/c2f/{name_of_task}.json", "w") as f:
    #     json.dump(results, f, indent=4)
