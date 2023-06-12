import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
import json
import math

scenes_names = ["Cutlery Block", "Wooden Block", "Bowl", "Teapot", "Purple Block"]
algo_names = ["PPO", "SAC", "BC"]
palette = sns.color_palette("tab10", n_colors=10)

### Format of data: array of [scene_index, amount_of_data, distance_errors, orientation_errors]
def smooth(scalars, weight=0.97):
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed

def create_graph(average_over_scenes, type_of_graph):

    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
    plt.rc('legend', fontsize=20)    # legend fontsize
    plt.rc('font', size=20)          # controls default text sizes

    if type_of_graph == "pos_error":
        plt.title("Position error")
        plt.ylabel("Position error (mm)")
    
    elif type_of_graph == "or_error":
        plt.title("Orientation error")
        plt.ylabel("Orientation error (degrees)")

def plot_line(x_data, y_data_mean, y_data_std, scene, label, colour):
    print("mean before smoothing", y_data_mean)
    print("std before smoothing", y_data_std)
    y_data_mean = np.array(smooth(y_data_mean, weight=0.99))
    y_data_std = np.array(smooth(y_data_std, weight=0.99))
    print("mean after smoothing", y_data_mean)
    print("std after smoothing", y_data_std)
    sns.lineplot(x=x_data, y=y_data_mean, color=colour, label=label)
    plt.fill_between(x_data, y_data_mean - y_data_std, y_data_mean + y_data_std, alpha=0.2, color=colour)

def complete_graph(average_over_scenes, name_of_graph):
    plt.xlabel("Amount of data")
    plt.legend(title="Scene" if not average_over_scenes else None)
    sns.despine()
    # limit x
    plt.xlim(0, 5000000)
    plt.savefig(f"final_results/final_graphs/{name_of_graph}.png")
    plt.show()

def plot_all_graphs(to_plot):
    algo_num, algo, x_data = to_plot
    print(algo, x_data)
    including_purple_block = [False] if algo == "bc" else [True, False] 
    for type_of_graph in ["pos_error", "or_error"]:
        for average_over_scenes in [True, False]:
            for include_purple_block in including_purple_block:

                if average_over_scenes:
                    average_text = 'average_over_scenes'
                else:
                    average_text = 'per_scene'
                if include_purple_block:
                    include_purple_block_text = ''
                    scenes = [0, 1, 2, 3, 4]
                else:
                    include_purple_block_text = '_without_purple_block'
                    scenes = [0, 1, 2, 3]

                # Create graph
                create_graph(average_over_scenes, type_of_graph)
                if average_over_scenes:
                    sac_data = np.load(f"{algo}_data_to_graph_{average_text}{include_purple_block_text}.npy", allow_pickle=True)

                    if type_of_graph == "pos_error":
                        plot_line(x_data, sac_data[0][0], sac_data[2][0], 0, label=algo_names[algo_num], colour=palette[algo_num])
                    elif type_of_graph == "or_error":
                        plot_line(x_data, sac_data[1][0], sac_data[3][0], 0, label=algo_names[algo_num], colour=palette[algo_num])

                else:
                    sac_data = np.load(f"{algo}_data_to_graph_per_scene.npy", allow_pickle=True)

                    for scene in scenes:
                        if type_of_graph == "pos_error":
                            plot_line(x_data, sac_data[0][scene], sac_data[2][scene], scene, label=scenes_names[scene], colour=palette[scene])
                        elif type_of_graph == "or_error":
                            plot_line(x_data, sac_data[1][scene], sac_data[3][scene], scene, label=scenes_names[scene], colour=palette[scene])

                complete_graph(average_over_scenes, f"{algo}_{type_of_graph}_{average_text}{include_purple_block_text}")

def plot_graph_with_algos(to_plot_algos, scenes_to_plot):
    for type_of_graph in ["pos_error", "or_error"]:
        for average_over_scenes in [True, False]:
            for include_purple_block in [True, False]:
                    create_graph(average_over_scenes, type_of_graph)
                    for to_plot in to_plot_algos:
                        algo_num, algo, x_data = to_plot
                        print(algo, x_data)

                        if average_over_scenes:
                            average_text = 'average_over_scenes'
                        else:
                            average_text = 'per_scene'
                        if include_purple_block:
                            include_purple_block_text = ''
                            scenes = [0, 1, 2, 3, 4]
                        else:
                            include_purple_block_text = '_without_purple_block'
                            scenes = [0, 1, 2, 3]

                        if algo == "bc":
                            scenes = [0, 1, 2, 3]

                        # Create graph
                        if average_over_scenes:
                            sac_data = np.load(f"{algo}_data_to_graph_{average_text}{include_purple_block_text}.npy", allow_pickle=True)

                            if type_of_graph == "pos_error":
                                plot_line(x_data, sac_data[0][0], sac_data[2][0], 0, label=algo_names[algo_num], colour=palette[algo_num])
                            elif type_of_graph == "or_error":
                                plot_line(x_data, sac_data[1][0], sac_data[3][0], 0, label=algo_names[algo_num], colour=palette[algo_num])

                        else:
                            sac_data = np.load(f"{algo}_data_to_graph_per_scene.npy", allow_pickle=True)

                            for scene in scenes_to_plot:
                                if type_of_graph == "pos_error":
                                    plot_line(x_data, sac_data[0][scene], sac_data[2][scene], scene, label=algo_names[algo_num] + ": " + scenes_names[scene], colour=palette[(algo_num*len(scenes_to_plot))+scene])
                                elif type_of_graph == "or_error":
                                    plot_line(x_data, sac_data[1][scene], sac_data[3][scene], scene, label=algo_names[algo_num] + ": " + scenes_names[scene], colour=palette[(algo_num*len(scenes_to_plot))+scene])

                    all_algo_names = ""
                    for to_plot in to_plot_algos:
                        algo_num, algo, x_data = to_plot
                        all_algo_names += algo_names[algo_num] + "_"
                    complete_graph(average_over_scenes, f"{all_algo_names}{type_of_graph}_{average_text}{include_purple_block_text}")

to_plot_ppo = [0, "ppo", [500000*i for i in range(1, 21)]]
to_plot_sac = [1, "sac", [500000*i for i in range(1, 21)]]
to_plot_bc = [2, "bc", [10000, 100000, 1000000]]
scenes_to_plot = [0, 1, 2, 3]
# average_over_scenes = True
# include_purple_block = False
# if average_over_scenes:
#     average_text = 'average_over_scenes'
# else:
#     average_text = 'per_scene'
# if include_purple_block:
#     include_purple_block_text = ''
#     scenes = [0, 1, 2, 3, 4]
# else:
#     include_purple_block_text = '_without_purple_block'
#     scenes = [0, 1, 2, 3]
# plot_all_graphs(to_plot_ppo)
# plot_all_graphs(to_plot_sac)
# plot_all_graphs(to_plot_bc)
plot_graph_with_algos([to_plot_ppo, to_plot_sac, to_plot_bc], scenes_to_plot)
