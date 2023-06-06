import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
import json

### Format of data: array of [scene_index, amount_of_data, distance_error, orientation_error]

def get_ppo_data(data_folder, type_of_graph):
    # Get all json from files in folder
    json_files = [file for file in os.listdir(data_folder) if file.endswith('.json')]

    # Read json files and make np array
    data = []

    for file in json_files:
        print("Reading file: ", file)
        with open(data_folder + file) as f:
            json_data = json.load(f)

            # Convert orientation error to degrees and distance error to millimeters
            for entry in json_data:
                entry['orientation_error'] = entry['orientation_error'] * 180 / np.pi  # Convert radians to degrees
                entry['distance_error'] = entry['distance_error'] * 1000  # Convert meters to millimeters

            json_data = json_data[0]
            data.append(np.array([json_data.get("scene_index"), json_data.get("seed"), json_data.get("amount_of_data"), json_data.get("distance_error"), json_data.get("orientation_error")]))
    data = np.array(data)

    groups = []
    # Sort data by scene index and amount of data
    data = sorted(data, key=lambda x: (x[0], x[2]))
    for key, group in groupby(data, lambda x: (x[0], x[2])):
        subarrs = list(group)

        if type_of_graph == "pos_error": 
            dist_avg = np.mean([subarr[3] for subarr in subarrs])
            groups.append([subarrs[0][0], subarrs[0][2], dist_avg])
        elif type_of_graph == "or_error":
            or_avg = np.mean([subarr[4] for subarr in subarrs])
            groups.append([subarrs[0][0], subarrs[0][2], or_avg])

    return groups

def create_graph(data, average_over_scenes, type, name_of_graph):

    palette = sns.color_palette("tab10", n_colors=6)

    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    plt.rc('font', size=16)          # controls default text sizes

    if type_of_graph == "pos_error":
        plt.title("Position error")
        plt.ylabel("Position error (m)")
    
    elif type_of_graph == "or_error":
        plt.title("Orientation error")
        plt.ylabel("Orientation error (degrees)")

    # print(data)
    data = np.array(data)
    # sort by scene index
    scenes = np.unique(data[:, 0])  # Get unique scene values

    # print(data)
    if average_over_scenes == False:
        for scene in scenes:
            scene= int(scene)
            scene_data = data[data[:, 0] == scene]  # Filter data for the current scene
            x = scene_data[:, 1]
            y = scene_data[:, 2]
            sns.lineplot(x=x, y=y, color=palette[scene], label=f"Scene {scene}")

    else:

        # Group data by amount of data
        data_to_plot_x = []
        data_to_plot_y = []
        data = sorted(data, key=lambda x: x[1])
        for key, group in groupby(data, lambda x: x[1]):
            subarrs = list(group)
            # Calculate mean of 2nd column
            print(subarrs)
            x = np.mean([subarr[1] for subarr in subarrs])
            y = np.mean([subarr[2] for subarr in subarrs])
            data_to_plot_x.append(x)
            data_to_plot_y.append(y)

        sns.lineplot(x=data_to_plot_x, y=data_to_plot_y, color=palette[0])

    plt.xlabel("Amount of data")
    plt.legend()
    sns.despine()
    plt.savefig(f"final_results/final_graphs/{name_of_graph}.png")
    plt.show()

algo = "ppo"
type_of_graph = "or_error"
data_folder = f"final_results/{algo}_final_results/"
avg_over_scenes = True
if avg_over_scenes:
    avg_text = 'average_over_scenes'
else:
    avg_text = 'per_scene'

ppo_data = get_ppo_data(data_folder, type_of_graph)

create_graph(data=ppo_data,
            average_over_scenes=avg_over_scenes,
            name_of_graph=f"{algo}_{avg_text}_{type_of_graph}",
            type=type_of_graph)