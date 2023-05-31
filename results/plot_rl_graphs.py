import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import make_interp_spline
import math

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

net_archs = [[32, 64, 128],
            [32, 64, 64, 128],
            [128, 128],
            [128, 128, 128],
            [64, 128, 256],
            [32, 64, 128, 256]]

def plot_tensorboard_graphs(base_directory, labels, name_of_graph, algo, tag_to_plot='rollout/ep_rew_mean'):
    # Create lists to store the data
    rollout_data = []
    ep_len_reward_data = []

    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=18)    # legend fontsize
    plt.rc('font', size=16)          # controls default text sizes

    # sns.set_style('dark')
    sns.despine()
    palette = sns.color_palette("tab10", n_colors=6)
    print("palette: ", palette)

    # Iterate over the subdirectories
    for label in labels:
        i, j, label = label
        colour_for_net_arch = palette[i]  # Assign color based on net_arch_i
        # Add label in legend for each net_arch
        if j == 0:
            plt.plot([], [], color=colour_for_net_arch, label=f'Network architecture: {net_archs[i]}')
        subdir_path = os.path.join(base_directory, label)
        subdir_path = os.path.join(subdir_path, f'{algo}_1/')

        # Find the events file in the subdirectory
        events_file = [f for f in os.listdir(subdir_path) if f.startswith('events.out')]

        if len(events_file) == 1:
            file_path = subdir_path + events_file[0]
        
            # Create an EventAccumulator and load the events
            accumulator = EventAccumulator(file_path)
            accumulator.Reload()

            # Get the scalar data from the events
            scalar_data = {}
            for tag in accumulator.Tags()['scalars']:
                scalar_data[tag] = accumulator.Scalars(tag)

            # print("scalar_data: ", scalar_data)
            # Plot the scalar data
            for tag, data in scalar_data.items():
                if tag == tag_to_plot:
                    steps = np.array([event.step for event in data])
                    values = np.array([event.value for event in data])

                    y_smooth = smooth(values, 0.99)

                    color = palette[i]  # Assign color based on net_arch_i
                    sns.lineplot(x=steps, y=y_smooth, color=color, errorbar=None)


    plt.xlabel('Timestep')
    plt.ylabel('Episode Length' if tag_to_plot == 'rollout/ep_len_mean' else 'Episode Reward')
    # plt.title('Episode Rewards')
    sns.despine()
    plt.legend()
    plt.savefig(name_of_graph)
    plt.show()


algo = 'SAC'
type_of_graph = 'rew'

# Set the directory containing the subdirectories with events files
base_directory = f'/vol/bitbucket/av1019/{algo}/tensorboard_logs/'

# Define the labels for the plots
labels = [[i, j, f'net_arch_{i}_scene_{j}'] for i in range(5) for j in range(6)]

# Plot the graphs
plot_tensorboard_graphs(base_directory=base_directory,
                        labels=labels,
                        name_of_graph=f'{algo}_tuning_graphs_ep_{type_of_graph}.png',
                        algo=algo,
                        tag_to_plot='rollout/ep_rew_mean' if type_of_graph == 'rew' else 'rollout/ep_len_mean')