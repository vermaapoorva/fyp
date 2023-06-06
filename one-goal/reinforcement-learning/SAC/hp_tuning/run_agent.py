from sac import train, run_model, get_number_of_parameters
import os
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import json

if __name__ == '__main__':

    # tuning_seed = 210423

    scenes = [["cutlery_block_scene.ttt", [-0.023, -0.08, 0.75, -3.140]],
        ["wooden_block_scene.ttt", [0.0843, -0.0254, 0.732, 1.100]],
        ["bowl_scene.ttt", [-0.074, -0.023, +0.7745, -2.915]],
        ["teapot_scene.ttt", [0.0573, -0.0254, 0.752, 2.871]],
        ["purple_block_scene.ttt", [-0.015, 0.012, 0.720, 1.71042]]]

    # If SAC directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/SAC"):
        os.makedirs("/vol/bitbucket/av1019/SAC")

    # If hyperparameters directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/SAC/hyperparameters"):
        os.makedirs("/vol/bitbucket/av1019/SAC/hyperparameters")

    # If values directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/SAC/hyperparameters/values"):
        os.makedirs("/vol/bitbucket/av1019/SAC/hyperparameters/values")

    for amount_of_data in range(1, 11):
        amount_of_data *= 1000000
        # amount_of_data -= 500000
        for final_seed in [210423]:
            for scene_num in range(len(scenes)):

                evaluation_file = f"/vol/bitbucket/av1019/SAC/final_models/results/final_results_{amount_of_data}_scene_{scene_num}_seed_{final_seed}.json"
                
                task_name = f"final_model_with_checkpoints_scene_{scene_num}_seed_{final_seed}"
                scene_file_name, bottleneck = scenes[scene_num]
                scene_name = scene_file_name.split(".")[0]

                if os.path.exists(evaluation_file):
                    print("Skipping, already evaluated")
                    continue
                    
                if not os.path.exists(f"/vol/bitbucket/av1019/SAC/logs/{task_name}/final_model_{scene_name}_{amount_of_data}_steps.zip"):
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

                distance_error, orientation_error, successful_episodes, percentage_of_successful_episodes = run_model(scene_file_name=scene_file_name,
                                                                                                                        bottleneck=bottleneck,
                                                                                                                        task_name=task_name,
                                                                                                                        num_of_runs=50,
                                                                                                                        amount_of_data=amount_of_data)

                results.append({"scene_index": scene_num,
                    "seed": final_seed,
                    "amount_of_data": amount_of_data,
                    "distance_error": distance_error,
                    "orientation_error": orientation_error,
                    "successful_episodes": successful_episodes,
                    "percentage_of_successful_episodes": percentage_of_successful_episodes})

                print(results)

                # Save results
                with open(evaluation_file, "w") as f:
                    json.dump(results, f, indent=4)

    # final_buffer_size = 250000
    # final_net_arch = [128, 128]

    # results = []
    # scene_num = 2

    # task_name = f"final_model_with_checkpoints_scene_{scene_num}_seed_{tuning_seed}"
    # scene_file_name, bottleneck = scenes[scene_num]

    # # # Save the hyperparameters, scene name and bottleneck to a file, create it if it doesn't exist
    # # with open(f"/vol/bitbucket/av1019/SAC/hyperparameters/values/{task_name}.txt", "w+") as f:
    # #     f.write(f"Scene: {scene_file_name}\n")
    # #     f.write(f"Bottleneck: {bottleneck}\n")
    # #     f.write(f"Net arch: {final_net_arch}\n")
    # #     f.write(f"Buffer size: {buffer_size}\n")

    # # Train the model
    # print(f"Training model on scene {scene_num} with seed {tuning_seed}")

    # train(scene_file_name=scene_file_name,
    #             bottleneck=bottleneck,
    #             seed=tuning_seed,
    #             hyperparameters={"net_arch": final_net_arch, "buffer_size": final_buffer_size},
    #             task_name=task_name)

    # print(f"Running model on scene {scene_num} with seed {tuning_seed}")
    # distance_error, orientation_error, successful_episodes, percentage_of_successful_episodes = run_model(scene_file_name=scene_file_name, bottleneck=bottleneck, task_name=task_name, num_of_runs=50)

    # results.append({"scene_index": scene_num,
    #     "distance_error": distance_error,
    #     "orientation_error": orientation_error,
    #     "successful_episodes": successful_episodes,
    #     "percentage_of_successful_episodes": percentage_of_successful_episodes})

    # print(results)

    # # Save results
    # with open(f"/vol/bitbucket/av1019/SAC/hyperparameters/results/{task_name}.json", "w") as f:
    #     json.dump(results, f, indent=4)