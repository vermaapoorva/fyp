import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import pandas as pd
import json
import math

AXIS_TEXT_SIZE = 24
GENERAL_TEXT_SIZE = 18

scenes_names = ["Cutlery Block", "Wooden Block", "Bowl", "Teapot", "Purple Block"]
scene_file_names = ["Cutlery_Block", "Wooden_Block", "Bowl", "Teapot", "Purple_Block"]
algo_names = ["PPO", "SAC", "BC", "DAgger"]
palette = sns.color_palette("tab10", n_colors=10)

### Format of data: array of [scene_index, amount_of_data, distance_errors, orientation_errors]
def smooth(scalars, weight=0):
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
    plt.rc('axes', titlesize=AXIS_TEXT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_TEXT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=GENERAL_TEXT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=GENERAL_TEXT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=GENERAL_TEXT_SIZE)    # legend fontsize
    plt.rc('font', size=GENERAL_TEXT_SIZE)          # controls default text sizes

    print("Type of error ", type_of_graph)
    if type_of_graph == "pos_error":
        print("pos error")
        # plt.title("Position error")
        plt.ylabel("Position error (mm)")
    
    else:
        print("or error")
        # plt.title("Orientation error")
        plt.ylabel("Orientation error (degrees)")

def plot_line(x_data, y_data_mean, y_data_std, scene, label, colour):
    y_data_mean = np.array(smooth(y_data_mean))
    y_data_std = np.array(smooth(y_data_std))
    sns.lineplot(x=x_data, y=y_data_mean, color=colour, label=label)
    plt.fill_between(x_data, y_data_mean - y_data_std, y_data_mean + y_data_std, alpha=0.2, color=colour)

def complete_graph(average_over_scenes, name_of_graph, clip_x_lower=0, clip_x_upper=10000000, y_min=None, y_max=None, log_scale=False):
    plt.xlabel("Number of images")
    plt.legend(title="Scene" if not average_over_scenes else None)
    sns.despine()
    # stretch y axis to make image taller
    
    
    # Add a thin dark grey dashed line at 0
    plt.axhline(y=0, color='#808080', linestyle='--', linewidth=1)
    print(clip_x_upper, clip_x_lower)
    # x axis to 10^6
    if y_min is not None and y_max is not None:
        plt.ylim(y_min-0.5, y_max+0.5)


    # plt.ylim(0, 15)

    if clip_x_upper == 1000000:
        plt.xscale('log')
    else:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(6, 6))
    plt.xlim(clip_x_lower, clip_x_upper)

    plt.savefig(f"final_results/final_graphs/{name_of_graph}.png")
    plt.show()

def plot_all_graphs(to_plot, clip_x_lower=0, clip_x_upper=10000000):
    algo_num, algo, x_data = to_plot
    print(algo, x_data)
    including_purple_block = [True, False]
    print("including_purple_block", including_purple_block)
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

                print(f"Include purple block: {include_purple_block}, text: {include_purple_block_text}")

                # Create graph
                create_graph(average_over_scenes, type_of_graph)
                if average_over_scenes:
                    datafile = f"data_to_plot/{algo}_data_to_graph_{average_text}{include_purple_block_text}.npy"
                    print(datafile)
                    sac_data = np.load(datafile, allow_pickle=True)

                    print(sac_data)

                    if type_of_graph == "pos_error":
                        plot_line(x_data, sac_data[0][0], sac_data[2][0], 0, label=algo_names[algo_num], colour=palette[algo_num])
                    elif type_of_graph == "or_error":
                        plot_line(x_data, sac_data[1][0], sac_data[3][0], 0, label=algo_names[algo_num], colour=palette[algo_num])

                else:
                    sac_data = np.load(f"data_to_plot/{algo}_data_to_graph_per_scene{include_purple_block_text}.npy", allow_pickle=True)

                    for scene in scenes:
                        if type_of_graph == "pos_error":
                            plot_line(x_data, sac_data[0][scene], sac_data[2][scene], scene, label=scenes_names[scene], colour=palette[scene])
                        elif type_of_graph == "or_error":
                            plot_line(x_data, sac_data[1][scene], sac_data[3][scene], scene, label=scenes_names[scene], colour=palette[scene])

                log_scale = False
                if algo == 'bc' or algo == 'dagger':
                    log_scale = True
                complete_graph(average_over_scenes, f"{algo}_{type_of_graph}_{average_text}{include_purple_block_text}_{clip_x_lower}_to_{clip_x_upper}", clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper, log_scale=log_scale)

def plot_graph_with_algos(to_plot_algos, clip_x_lower=0, clip_x_upper=10000000):
    including_purple_block = [True, False]
    y_min = 100000000
    y_max = -100000000
    # for algos in to_plot_algos:
        # if algos[1] == 'bc':
        #     print("skipping purple block")
        #     including_purple_block = [False]
    for type_of_graph in ["pos_error", "or_error"]:
        for include_purple_block in including_purple_block:
                create_graph(True, type_of_graph)
                for to_plot in to_plot_algos:
                    algo_num, algo, x_data = to_plot
                    print(algo, x_data)

                    average_text = 'average_over_scenes'
                    if include_purple_block:
                        include_purple_block_text = ''
                        scenes = [0, 1, 2, 3, 4]
                    else:
                        include_purple_block_text = '_without_purple_block'
                        scenes = [0, 1, 2, 3]

                    # Create graph
                    sac_data = np.load(f"data_to_plot/{algo}_data_to_graph_{average_text}{include_purple_block_text}.npy", allow_pickle=True)

                    if type_of_graph == "pos_error":
                        plot_line(x_data, sac_data[0][0], sac_data[2][0], 0, label=algo_names[algo_num], colour=palette[algo_num])
                    elif type_of_graph == "or_error":
                        plot_line(x_data, sac_data[1][0], sac_data[3][0], 0, label=algo_names[algo_num], colour=palette[algo_num])

                    # get indices greater than x lower clip and lower than x upper clip
                    x_lower_clip_index = [i for i in range(len(x_data)) if x_data[i] > clip_x_lower][0]
                    x_upper_clip_index = [i for i in range(len(x_data)) if x_data[i] < clip_x_upper][-1]
                    print(x_lower_clip_index, x_upper_clip_index)
                    if x_lower_clip_index == x_upper_clip_index:
                        x_upper_clip_index += 1
                    # check y min and max within x range and make sure they're visible 
                    if type_of_graph == "pos_error":
                        y_min = min(y_min, np.min(sac_data[0][0][x_lower_clip_index:x_upper_clip_index]))
                        y_max = max(y_max, np.max(sac_data[0][0][x_lower_clip_index:x_upper_clip_index]))
                    elif type_of_graph == "or_error":
                        y_min = min(y_min, np.min(sac_data[1][0][x_lower_clip_index:x_upper_clip_index]))
                        y_max = max(y_max, np.max(sac_data[1][0][x_lower_clip_index:x_upper_clip_index]))

                all_algo_names = ""
                for to_plot in to_plot_algos:
                    algo_num, algo, x_data = to_plot
                    all_algo_names += algo_names[algo_num] + "_"
                final_graph_name = f"{all_algo_names}{type_of_graph}_{average_text}{include_purple_block_text}_{clip_x_lower}_to_{clip_x_upper}"
                print(f"Graph: {final_graph_name}")
                
                complete_graph(True, final_graph_name, clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
                            #    , y_min=y_min, y_max=y_max)

def plot_graph_with_scenes_and_algos(to_plot_algos, scenes_to_plot, single_graph, clip_x_lower=0, clip_x_upper=10000000):
    for type_of_graph in ["pos_error", "or_error"]:
        if single_graph:
            create_graph(False, type_of_graph)
        for scene in scenes_to_plot:
            if not single_graph:
                create_graph(False, type_of_graph)
            for algo_i, to_plot in enumerate(to_plot_algos):
                algo_num, algo, x_data = to_plot

                if scene != 4:
                    include_purple_block_text = '_without_purple_block'
                else:
                    include_purple_block_text = ''

                sac_data = np.load(f"data_to_plot/{algo}_data_to_graph_per_scene{include_purple_block_text}.npy", allow_pickle=True)

                colour = palette[algo_num]
                if single_graph:
                    colour = palette[len(scenes_to_plot)*algo_i + scene]

                if type_of_graph == "pos_error":
                    plot_line(x_data, sac_data[0][scene], sac_data[2][scene], scene, label=algo_names[algo_num] + ": " + scenes_names[scene], colour=colour)
                elif type_of_graph == "or_error":
                    plot_line(x_data, sac_data[1][scene], sac_data[3][scene], scene, label=algo_names[algo_num] + ": " + scenes_names[scene], colour=colour)

            all_algo_names = ""
            log_scale=False
            for to_plot in to_plot_algos:
                algo_num, algo, x_data = to_plot
                all_algo_names += algo_names[algo_num] + "_"
            final_graph_name = f"{all_algo_names}{type_of_graph}_{scene_file_names[scene]}"
            print(f"Graph: {final_graph_name}")
            if not single_graph:
                if clip_x_upper <= 1000000:
                    log_scale=True
                complete_graph(False, final_graph_name, clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper, log_scale=log_scale)
        if single_graph:
            scene_names_plotted = ""
            for scene in scenes_to_plot:
                scene_names_plotted += scene_file_names[scene] + "_"
            if clip_x_upper <= 1000000:
                log_scale=True
            complete_graph(False, f"{all_algo_names}{type_of_graph}_{scene_names_plotted}", clip_x_lower, clip_x_upper)

def plot_graph_per_scene_average_across_algos(to_plot_scenes, to_plot_algos, clip_x_lower=0, clip_x_upper=10000000):
    # include_purple_block_text = '_without_purple_block'
    include_purple_block_text = ''
    x_datas = [x_data for _, _, x_data in to_plot_algos]
    final_x_data = []

    for val in x_datas[0]:
        include_val = True
        for x_data in x_datas:
            if val not in x_data:
                include_val = False
                break
        if include_val:
            final_x_data.append(val)

    print(final_x_data)

    for type_of_graph in ['pos_error', 'or_error']:
        create_graph(False, type_of_graph)

        for scene in scenes_to_plot:

            all_y_data_mean_pos = []
            all_y_data_std_pos = []
            all_y_data_mean_or = []
            all_y_data_std_or = []

            for to_plot_algo in to_plot_algos:
                algo_num, algo, x_data = to_plot_algo
                print(scene, algo)
                if scene==4 and algo=='bc':
                    continue

                algo_y_data = np.load(f"data_to_plot/{algo}_data_to_graph_per_scene{include_purple_block_text}.npy", allow_pickle=True)

                algo_all_mean_pos = algo_y_data[0][scene]
                algo_all_std_pos = algo_y_data[1][scene]
                algo_all_mean_or = algo_y_data[2][scene]
                algo_all_std_or = algo_y_data[3][scene]

                to_plot_y_mean_pos = []
                to_plot_y_std_pos = []
                to_plot_y_mean_or = []
                to_plot_y_std_or = []

                for index, val in enumerate(x_data):
                    if val in final_x_data:
                        to_plot_y_mean_pos.append(algo_all_mean_pos[index])
                        to_plot_y_std_pos.append(algo_all_std_pos[index])
                        to_plot_y_mean_or.append(algo_all_mean_or[index])
                        to_plot_y_std_or.append(algo_all_std_or[index])

                all_y_data_mean_pos.append(np.array(to_plot_y_mean_pos))
                all_y_data_std_pos.append(np.array(to_plot_y_std_pos))
                all_y_data_mean_or.append(np.array(to_plot_y_mean_or))
                all_y_data_std_or.append(np.array(to_plot_y_std_or))

            all_y_data_mean_pos = np.array(all_y_data_mean_pos)
            all_y_data_std_pos = np.array(all_y_data_std_pos)
            all_y_data_mean_or = np.array(all_y_data_mean_or)
            all_y_data_std_or = np.array(all_y_data_std_or)

            mean_pos_to_plot = np.mean(all_y_data_mean_pos, axis=0)
            std_pos_to_plot = np.mean(all_y_data_std_pos, axis=0)
            mean_or_to_plot = np.mean(all_y_data_mean_or, axis=0)
            std_or_to_plot = np.mean(all_y_data_std_or, axis=0)

            if type_of_graph == "pos_error":
                plot_line(final_x_data, mean_pos_to_plot, std_pos_to_plot, scene, label=scenes_names[scene], colour=palette[scene])
            elif type_of_graph == "or_error":
                plot_line(final_x_data, mean_or_to_plot, std_or_to_plot, scene, label=scenes_names[scene], colour=palette[scene])
        
        all_algo_names = ""
        for to_plot in to_plot_algos:
            algo_num, algo, x_data = to_plot
            all_algo_names += algo_names[algo_num] + "_"
        final_graph_name = f"average_over_{all_algo_names}{type_of_graph}_per_scene_{include_purple_block_text}"
        clip_y_lower = None
        clip_y_upper = None
        if clip_x_lower and clip_x_upper:
            final_graph_name += f'{clip_x_lower}_to_{clip_x_upper}'
            if type_of_graph == 'pos_error':
                clip_y_lower = -2
                clip_y_upper = 10
            else:
                clip_y_lower = -4
                clip_y_upper = 10
            
        complete_graph(False, final_graph_name, clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
        # , y_min=clip_y_lower, y_max=clip_y_upper)

to_plot_ppo = [0, "ppo", [10000, 100000] + [500000*i for i in range(1, 21)]]
to_plot_sac = [1, "sac", [10000, 100000] + [500000*i for i in range(1, 21)]]
to_plot_bc = [2, "bc", [10000, 100000, 1000000, 5000000, 10000000]]
to_plot_dagger = [3, "dagger", [10000, 100000, 1000000]]
scenes_to_plot = [4]
single_graph = True
clip_x_lower = 0
clip_x_upper = 10000000

# plot_all_graphs(to_plot_ppo)
# plot_all_graphs(to_plot_sac, clip_x_lower, clip_x_upper)
# plot_all_graphs(to_plot_bc, clip_x_lower, clip_x_upper)
# plot_all_graphs(to_plot_dagger, clip_x_lower, clip_x_upper)
# plot_graph_with_algos([to_plot_ppo, to_plot_sac], clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
# plot_graph_with_algos([to_plot_bc, to_plot_dagger], clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
plot_graph_with_scenes_and_algos(to_plot_algos=[to_plot_ppo, to_plot_sac, to_plot_bc], scenes_to_plot=scenes_to_plot, single_graph=single_graph)
# plot_graph_with_scenes_and_algos(to_plot_algos=[to_plot_sac], scenes_to_plot=scenes_to_plot, single_graph=single_graph)
# plot_graph_with_scenes_and_algos(to_plot_algos=[to_plot_bc], scenes_to_plot=scenes_to_plot, single_graph=single_graph, clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
# plot_graph_with_scenes_and_algos(to_plot_algos=[to_plot_dagger], scenes_to_plot=scenes_to_plot, single_graph=single_graph, clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)
# plot_graph_per_scene_average_across_algos(to_plot_scenes=scenes_to_plot, to_plot_algos=[to_plot_ppo, to_plot_sac, to_plot_bc, to_plot_dagger], clip_x_lower=clip_x_lower, clip_x_upper=clip_x_upper)