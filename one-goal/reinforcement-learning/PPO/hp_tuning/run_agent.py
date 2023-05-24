from reinforcement_learning import train, run_model, get_number_of_parameters
import os
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

if __name__ == '__main__':

    hyperparameters = [{"net_arch": [32, 48, 64, 128], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [32, 48, 64, 128], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [32, 48, 64, 128], "batch_size": 8192, "n_steps": 4096}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [32, 64, 128], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [32, 64, 128], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [32, 64, 128], "batch_size": 8192, "n_steps": 4096}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [32, 64, 64, 128], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [32, 64, 64, 128], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [32, 64, 64, 128], "batch_size": 8192, "n_steps": 4096}, # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16
                        {"net_arch": [64, 128, 64], "batch_size": 4096, "n_steps": 2048}, # rollout buffer size: 2048*16 = 32768, updates: 32768/4096 = 8
                        {"net_arch": [64, 128, 64], "batch_size": 8192, "n_steps": 2048}, # rollout buffer size: 2048*16*2 = 65536, updates: 65536/8192 = 8
                        {"net_arch": [64, 128, 64], "batch_size": 8192, "n_steps": 4096}] # rollout buffer size: 4096*16*2 = 131072, updates: 131072/8192 = 16

    scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
                ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
                ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
                ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
                ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
                ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]

    # If PPO directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO"):
        os.makedirs("/vol/bitbucket/av1019/PPO")

    # If hyperparameters directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO/hyperparameters"):
        os.makedirs("/vol/bitbucket/av1019/PPO/hyperparameters")

    # If values directory doesn't exist, create it
    if not os.path.exists("/vol/bitbucket/av1019/PPO/hyperparameters/values"):
        os.makedirs("/vol/bitbucket/av1019/PPO/hyperparameters/values")

    hp_indexes = [7]

    scene_indexes = [2]
    # scene_indexes = [3, 4, 5]
    env = Monitor(gym.make("RobotEnv-v2", file_name=scenes[0][0], bottleneck=scenes[0][1], headless=True, image_size=64, sleep=0), "logs_arch/")

    net_archs = [[32, 48, 64, 128], [32, 64, 128], [32, 64, 64, 128], [64, 128, 64]]
    for net_arch in net_archs:
        params = get_number_of_parameters(net_arch, env)
        print(f"Net arch: {net_arch}, params: {params}")

    # for i in hp_indexes:
    #     for scene_num in scene_indexes:
    #         scene_file_name, bottleneck = scenes[scene_num]
    #         # Save the hyperparameters, scene name and bottleneck to a file, create it if it doesn't exist
    #         with open(f"/vol/bitbucket/av1019/PPO/hyperparameters/values/hps_{i}_scene_{scene_num}.txt", "w+") as f:
    #             f.write(f"Scene: {scene_file_name}\n")
    #             f.write(f"Bottleneck: {bottleneck}\n")
    #             f.write(f"Hyperparameters: {hyperparameters[i]}\n")

    #         # Train the model
    #         print(f"Training model {i} on scene {scene_num} with hyperparameters {hyperparameters[i]}")
    #         train(scene_file_name, bottleneck, hyperparameters[i], i, scene_num)

    # run_model()