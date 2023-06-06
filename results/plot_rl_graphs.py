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
scenes = ["1000", "10000", "100000", "250000"]

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

def get_data_from_tensorboard(labels, base_directory, algo, tag_to_plot):
    steps_values_to_plot = []
    # Iterate over the subdirectories
    for label in labels:
        scene_i, seed_j, label = label
        subdir_path = os.path.join(base_directory, label)
        subdir_path = os.path.join(subdir_path, f'{algo}_1/')
    
        # if path does not exist, skip
        if not os.path.exists(subdir_path):
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

    return steps_values_to_plot

def group_func(x, avg_over_scenes):
    if avg_over_scenes:
        return 1
    else:
        return x["scene"]

def plot_tensorboard_graphs(base_directory, labels, name_of_graph, algo, tag_to_plot='rollout/ep_rew_mean', avg_over_scenes=False):
    # Create lists to store the data
    rollout_data = []
    ep_len_reward_data = []

    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    plt.rc('font', size=18)          # controls default text sizes

    # sns.set_style('dark')
    sns.despine()
    palette = sns.color_palette("tab10", n_colors=6)
    print("palette: ", palette)

    steps_values_to_plot = get_data_from_tensorboard(labels, base_directory, algo, tag_to_plot)

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

            # Get index of step where value is > 10M
            max_index = next((i for i in range(0, len(steps)) if steps[i] > 1500000), None)
            if max_index is not None:
                steps = steps[:max_index]
                values = values[:max_index]

            # Find indices of duplicates in steps
            _, unique_indices = np.unique(steps, return_index=True)
            duplicate_indices = np.setdiff1d(np.arange(len(steps)), unique_indices)

            # Remove duplicates from steps and values arrays
            steps = np.delete(steps, duplicate_indices)
            values = np.delete(values, duplicate_indices)
            print("Steps after removing duplicates:", steps)

            # if len(steps) < 200:
            #     diff = 200 - len(steps)
            #     extras = steps[-diff:]
            #     steps = np.append(steps, extras)
                
            #     diff = 200 - len(values)
            #     extras = values[-diff:]
            #     values = np.append(values, extras)

            # Calculate the step size
            skip = int(np.round(len(steps) / 200))

            # Remove elements at regular intervals
            steps = steps[::skip]
            values = values[::skip]

            if len(steps) < 200:
                diff = 200 - len(steps)
                extras = steps[-diff:]
                steps = np.append(steps, extras)
                
                diff = 200 - len(values)
                extras = values[-diff:]
                values = np.append(values, extras)

            all_steps.append(steps)
            all_values.append(values)

        # if scene_i == 4 or group_i == 4:

        # diff = len(all_steps[-2]) - len(all_steps[-1])
        # if diff > 0:
        #     extras = all_steps[-1][-diff:]
        #     # all_steps[1].append(extras)
        #     all_steps[-1] = np.append(all_steps[-1], extras)

        # diff = len(all_values[-2]) - len(all_values[-1])
        # if diff > 0:
        #     extras = all_values[-1][-diff:]
        #     all_values[-1] = np.append(all_values[-1], extras)

        # shortest_index = np.argmin([len(arr) for arr in all_steps])
        # shortest_array = all_steps[shortest_index]
        
        # print(len(shortest_array))
        # print(shortest_array)

        # common_elements = set(all_steps[0]).intersection(*all_steps[1:])
        # print(common_elements)

        # Clip arrays to length of shortest length array
        min_length = min([len(x) for x in all_steps])
        all_steps = [x[:min_length] for x in all_steps]
        all_values = [x[:min_length] for x in all_values]

        # smooth each elem in values arrays
        all_values_smoothed = np.array([smooth(x, 0.99) for x in all_values])

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
            label=None
        else:
            label=scenes[scene_i]

        sns.lineplot(data=data_to_plot, x='x', y='vals', color=colour, errorbar=None, label=label)

        scene_i += 1

    plt.xlabel('Timestep')
    plt.ylabel('Episode Length' if tag_to_plot == 'rollout/ep_len_mean' else 'Episode Reward')
    # plt.title('Episode Rewards')
    sns.despine()
    plt.legend(title="Buffer size" if not avg_over_scenes else None)
    # Clip the graph to the first 10 million timesteps
    plt.xlim(0, 1500000)
    plt.savefig(name_of_graph)
    plt.show()


algo = 'SAC'
type_of_graph = 'rew'
avg_over_scenes = False
if avg_over_scenes:
    avg_text = 'average_over_scenes'
else:
    avg_text = 'per_scene'

# Set the directory containing the subdirectories with events files
base_directory = f'/vol/bitbucket/av1019/{algo}/tensorboard_logs/'

# Define the labels for the plots
# labels = [[i, j, f'final_model_with_checkpoints_scene_{i}_seed_{j}'] for i in range(4) for j in [1019, 2603, 210423]]
labels = [[i, j, f'buffer_size_{i}_scene_{j}'] for i in [1000, 10000, 100000, 250000] for j in range(6)]

# Plot the graphs
plot_tensorboard_graphs(base_directory=base_directory,
                        labels=labels,
                        name_of_graph=f'{algo}_buffer_size_tuning_{avg_text}_{type_of_graph}.png',
                        algo=algo,
                        tag_to_plot='rollout/ep_rew_mean' if type_of_graph == 'rew' else 'rollout/ep_len_mean',
                        avg_over_scenes=avg_over_scenes)