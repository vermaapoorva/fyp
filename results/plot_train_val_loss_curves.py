import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

palette = sns.color_palette("tab10", n_colors=2)

def smooth(scalars, weight=0.9):
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

def create_graph():

    plt.figure(figsize=(15, 12), tight_layout=True)
    plt.rc('axes', titlesize=32)     # fontsize of the axes title
    plt.rc('axes', labelsize=32)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=28)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=28)    # fontsize of the tick labels
    plt.rc('legend', fontsize=32)    # legend fontsize
    plt.rc('font', size=32)          # controls default text sizes

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

def complete_graph(name_of_graph):
    # plt.legend(title="Loss")
    sns.despine()
    plt.ylim(0, 0.003)
    plt.savefig(f"{name_of_graph}.png")

    # plt.show()

scenes = ["cutlery_block", "wooden_block", "bowl", "teapot"]
amounts_of_data = [10000, 100000, 1000000, 5000000, 10000000]
network_losses = 'train_val_losses/bc/'

for scene in scenes:
    for amount_of_data in amounts_of_data:
        create_graph()
        data_path = f"{network_losses}{amount_of_data}_{scene}_scene_losses.npz"
        img_path = f"loss_curves/{amount_of_data}_{scene}_loss_curve"

        # if datapath doesnt exist, skip
        if os.path.exists(data_path) == False:
            continue

        # Load data from the NPZ file
        data = np.load(data_path)

        # Extract the relevant arrays
        training_losses = data['training_losses']
        validation_losses = data['validation_losses']
        epochs = data['epochs']

        if len(training_losses) != len(validation_losses) or len(training_losses) != len(epochs):
            print("length of training_losses", len(training_losses))
            print("length of validation_losses", len(validation_losses))
            print("length of epochs", len(epochs))

            length = len(training_losses)
            epochs = epochs[-length:]

            print("length of epochs", len(epochs))

        sns.lineplot(x=epochs, y=smooth(training_losses), label="Training Loss", color=palette[0])
        sns.lineplot(x=epochs, y=smooth(validation_losses), label="Validation Loss", color=palette[1])
        
        complete_graph(img_path)

        # # Crop graph to max validation_loss * 3
        # max_validation_loss = max(validation_losses)
        # plt.ylim(0, max_validation_loss * 3)

        # # Save the plot as an image file
        # plt.savefig(img_path)

        # # Display the plot
        # plt.show()