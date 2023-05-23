from train import ImageToPoseTrainerCoarse
from network import ImageToPoseNetworkCoarse
from evaluate import run_model

import gymnasium as gym
import robot_env
from tqdm import trange
import numpy as np
import torch

scenes = [["pitcher_scene.ttt", [0.05, 0.001, 0.78, 3.056]],
            ["twist_shape_scene.ttt", [-0.011, -0.023, 0.65, 1.616]],
            ["easter_basket_teal.ttt", [-0.045, 0.072, 0.712, 2.568]],
            ["white_bead_mug.ttt", [-0.043, -0.002, 0.718, -0.538]],
            ["frying_pan_scene.ttt", [0.100, 0.005, 0.675, -2.723]],
            ["milk_frother_scene.ttt", [0.020, -0.025, 0.728, -0.868]]]    

scene_index = 0

or_coeffs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
returns = []
distance_errors = []
orientation_errors = []
results = []

for or_coeff in or_coeffs:

    name_of_task = f"orientation_coeff_{or_coeff}"

    trainer = ImageToPoseTrainerCoarse(task_name=name_of_task)
    trainer.train()

    model_return, distance_error, orientation_error = run_model(task_name=name_of_task, scene_name=scenes[scene_index][0], bottleneck=scenes[scene_index][1], num_of_runs=5)

    results.append({"or_coeff": or_coeff, "model_return": model_return, "distance_error": distance_error, "orientation_error": orientation_error})

    print(f"or_coeff: {or_coeff}, model_return: {model_return}, distance_error: {distance_error}, orientation_error: {orientation_error}")

# Sort results by model_return
results = sorted(results, key=lambda k: k['model_return'], reverse=True)
print(results)