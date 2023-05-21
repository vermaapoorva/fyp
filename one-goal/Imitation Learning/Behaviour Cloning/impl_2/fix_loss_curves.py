import pickle
import matplotlib.pyplot as plt

hyperparameter_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
scene_indexes = [0, 1]

for hyperparameter_index in hyperparameter_indexes:
    for scene_index in scene_indexes:
        MODEL_INDEX = f"epoch_lr_batch_{hyperparameter_index}_object_{scene_index}"

        # Load the training history from the pickle file
        file_path = '/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/training_history/model_' + MODEL_INDEX + '.pkl'
        try:
            with open(file_path, 'rb') as file:
                history = pickle.load(file)
        except FileNotFoundError:
            continue

        # Plot the training and validation loss
        plt.plot(history['loss'][2:])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')

        # Set the y-axis limits
        y_min = min(min(history['loss']), min(history['val_loss']))
        plt.ylim(bottom=y_min - 0.001, top=0.102)

        # Save the plot to the 'losses' folder with the model index in the name
        plt.savefig('losses/loss_' + str(MODEL_INDEX) + '.png')
        plt.clf()
