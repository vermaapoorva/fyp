from reinforcement_learning import train, run_model, get_number_of_parameters
import os
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import json

if __name__ == '__main__':

    # If PPO directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO"):
        os.makedirs("/vol/bitbucket/av1019/PPO")

    # If hyperparameters directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO/final_models"):
        os.makedirs("/vol/bitbucket/av1019/PPO/final_models")

    # If values directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO/final_models/results"):
        os.makedirs("/vol/bitbucket/av1019/PPO/final_models/results")

    final_hyperparameters = {"net_arch": [128, 128], "batch_size": 4096, "n_steps": 2048}

    scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
        ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
        ["bowl_scene.ttt", [-0.074, -0.023, +0.7745, -2.915]],
        ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]],
        ["purple_block_scene.ttt", [-0.015, 0.012, 0.720, 1.71042]]]

    # final_seed = 1019

    # scene_num = 0

    # run for 1M, 2M, 3M, 4M, 5M, 6M, 7M, 8M, 9M, 10M
    for amount_of_data in range(1, 11):
        amount_of_data *= 1000000
        amount_of_data -= 500000
        for final_seed in [1019, 2603, 210423]:
            for scene_num in range(len(scenes)):

                evaluation_file = f"/vol/bitbucket/av1019/PPO/final_models/results/actual_final_results_{amount_of_data}_scene_{scene_num}_seed_{final_seed}.json"
                
                task_name = f"final_model_scene_{scene_num}_seed_{final_seed}"
                scene_file_name, bottleneck = scenes[scene_num]
                scene_name = scene_file_name.split(".")[0]

                if os.path.exists(evaluation_file):
                    print("Skipping, already evaluated")
                    continue
                    
                if not os.path.exists(f"/vol/bitbucket/av1019/PPO/logs/{task_name}/final_model_{scene_name}_{amount_of_data}_steps.zip"):
                    print(f"Skipping, model not trained, scene: {scene_num}, seed: {final_seed}, data: {amount_of_data}")
                    continue

                results = []

                # Train the model
                # print(f"Training final model for scene {scene_num} with seed {final_seed}")

                # train(scene_file_name=scene_file_name,
                #             bottleneck=bottleneck,
                #             seed=final_seed,
                #             hyperparameters=final_hyperparameters,
                #             task_name=task_name)

                print(f"Running final model for scene {scene_num} with seed {final_seed} and data {amount_of_data/1000000}M")

                distance_errors, orientation_errors, successful_episodes, percentage_of_successful_episodes = run_model(scene_file_name=scene_file_name,
                                                                                                                        bottleneck=bottleneck,
                                                                                                                        task_name=task_name,
                                                                                                                        num_of_runs=50,
                                                                                                                        amount_of_data=amount_of_data)

                results.append({"scene_index": scene_num,
                    "seed": final_seed,
                    "amount_of_data": amount_of_data,
                    "distance_errors": distance_errors,
                    "orientation_errors": orientation_errors,
                    "successful_episodes": successful_episodes,
                    "percentage_of_successful_episodes": percentage_of_successful_episodes})

                print(results)

                # Save results
                with open(evaluation_file, "w") as f:
                    json.dump(results, f, indent=4)