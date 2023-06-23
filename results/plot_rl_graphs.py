import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import make_interp_spline
import math
from itertools import groupby
import pandas as pd

scenes = ["Cutlery Block", "Wooden Block", "Bowl", "Teapot", "Purple Block"]
algo_names = ["PPO", "SAC"]

AXIS_TEXT_SIZE = 28
GENERAL_TEXT_SIZE = 28

palette = sns.color_palette("tab10", n_colors=6)

def smooth(scalars, weight=0.8):
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

def get_data_from_tensorboard(labels, base_directory, algo, tag_to_plot):
    steps_values_to_plot = []
    # Iterate over the subdirectories
    for label in labels:
        print("label: ", label)
        scene_i, seed_j, label = label
        subdir_path = os.path.join(base_directory, label)
        subdir_path = os.path.join(subdir_path, f'{algo}_1/')
    
        # if path does not exist, skip
        if not os.path.exists(subdir_path):
            print("Skipping path: ", subdir_path)
            continue

        # Create an EventAccumulator and load the events
        print(f"Loading events from {subdir_path}")
        accumulator = EventAccumulator(subdir_path, purge_orphaned_data=True)
        accumulator.Reload()

        # Get the scalar data from the events
        scalar_data = {}
        for tag in accumulator.Tags()['scalars']:
            scalar_data[tag] = accumulator.Scalars(tag)

        # Plot the scalar data
        for tag, data in scalar_data.items():
            if tag == tag_to_plot:
                steps = np.array([event.step for event in data])
                values = np.array([event.value for event in data])

                # If max value in steps < 10M, skip
                # if np.max(steps) < 10000000:
                #     continue
                
                decreasing_index = next((i for i in range(1, len(steps)) if steps[i] < steps[i-1]), None)

                if decreasing_index is not None:
                    # Find first element greater than the value at decreasing_index
                    reset_num = steps[decreasing_index]
                    increasing_index = next((i for i in range(0, len(steps)) if steps[i] > reset_num), None)
                    # remove section between increasing index and decreasing index
                    steps = np.concatenate((steps[:increasing_index], steps[decreasing_index:]))
                    values = np.concatenate((values[:increasing_index], values[decreasing_index:]))

                steps_values_to_plot.append({"scene": scene_i, "seed": seed_j, "steps": steps, "values": values})

                print(f"Added data for {label}")

    # Save steps to npy file
    type_of_plot = "rew" if "rew" in tag_to_plot else "len"
    np.save(f"{algo}_{type_of_plot}_data.npy", np.array(steps_values_to_plot))

    return steps_values_to_plot

def group_func(x, avg_over_scenes):
    if avg_over_scenes:
        return 1
    else:
        return x["scene"]

def create_graph():
    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=AXIS_TEXT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_TEXT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=GENERAL_TEXT_SIZE-4)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=GENERAL_TEXT_SIZE-4)    # fontsize of the tick labels
    plt.rc('legend', fontsize=GENERAL_TEXT_SIZE)    # legend fontsize
    plt.rc('font', size=GENERAL_TEXT_SIZE)          # controls default text sizes

    # sns.set_style('dark')
    sns.despine()

def plot_tensorboard_graphs(avg_over_scenes=False, steps_values_to_plot=None, algo=None, algo_num=0):

    scene_i = 0
    # Plot the data average over seeds
    for key, group in groupby(steps_values_to_plot, lambda x: group_func(x, avg_over_scenes)):
        colour = palette[scene_i]
        print("key: ", key)
        all_steps = []
        all_values = []
        for group_i, g in enumerate(group):
            steps = g["steps"]
            values = g["values"]

            all_steps.append(steps)
            all_values.append(values)

        # Clip arrays to length of shortest length array
        min_length = min([len(x) for x in all_steps])
        all_steps = [x[:min_length] for x in all_steps]
        all_values = [x[:min_length] for x in all_values]

        # smooth each elem in values arrays
        all_values_smoothed = np.array([smooth(x) for x in all_values])

        all_data = all_steps[0:1]
        all_data = np.append(all_data, all_values_smoothed, axis=0)

        all_data = np.transpose(all_data)
        # Create a longform dataframe
        columns = ['x']
        for i in range(0, len(all_data[0])-1):
            columns.append(str(i))
        data_to_plot = pd.DataFrame(data=all_data, columns=columns)
        data_to_plot = data_to_plot.melt('x', var_name='cols', value_name='vals')

        if avg_over_scenes:
            label=algo_names[algo_num]
            colour = palette[algo_num]
        else:
            label=scenes[scene_i]

        sns.lineplot(data=data_to_plot, x='x', y='vals', color=colour, errorbar='sd', label=label)

        scene_i += 1

def complete_graph(tag_to_plot, name_of_graph):

    plt.xlabel('Timestep')
    plt.ylabel('Episode Length' if tag_to_plot == 'rollout/ep_len_mean' else 'Episode Reward')
    # plt.title('Episode Rewards')
    sns.despine()
    # plt.legend(title="Network architecture" if not avg_over_scenes else None)
    # Clip the graph to the first 10 million timesteps
    plt.xlim(0, 2000000)
    # legent bottom right
    # plt.legend(loc='lower right')
    plt.savefig(name_of_graph)
    plt.show()

# labels = [[i, j, f'final_model_scene_{i}_seed_{j}'] for i in range(5) for j in [1019, 2603, 210423]]
# labels = [[i, j, f'final_model_with_checkpoints_scene_{i}_seed_{j}'] for i in range(5) for j in [1019, 2603, 210423]]
# Set the directory containing the subdirectories with events files
# base_directory = f'/home/apoorva/Documents/FYP/results/rl_tensorboard_logs/{algo}/'

# get_data_from_tensorboard(labels, base_directory, algo, tag_to_plot='rollout/ep_len_mean' if type_of_graph == 'len' else 'rollout/ep_rew_mean')

algos = ['SAC']
type_of_graph = 'rew'
avg_over_scenes = False
include_purple_block = True
if avg_over_scenes:
    avg_text = 'average_over_scenes'
else:
    avg_text = 'per_scene'
if include_purple_block:
    purple_text = ''
else:
    purple_text = '_without_purple_block'

# Plot the graphs
create_graph()

for algo_num, algo in enumerate(algos):

    # Read steps_values_to_plot from file
    with open(f"{algo}_{type_of_graph}_data{purple_text}.npy", "rb") as f:
        steps_values_to_plot = np.load(f, allow_pickle=True)

    # make it a list
    steps_values_to_plot = steps_values_to_plot.tolist()

    plot_tensorboard_graphs(avg_over_scenes=avg_over_scenes,
                            steps_values_to_plot=steps_values_to_plot,
                            algo=algo,
                            algo_num=algo_num)

algo_string = ''
for algo in algos:
    algo_string += f'{algo}_'

complete_graph(tag_to_plot='rollout/ep_rew_mean' if type_of_graph == 'rew' else 'rollout/ep_len_mean',
                name_of_graph=f'{algo_string}_final_2M_{avg_text}_{type_of_graph}.png')