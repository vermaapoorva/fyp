import numpy as np
import matplotlib.pyplot as plt


task_name = 'original_hp_results_scene_0'
bitbucket = '/vol/bitbucket/av1019/dagger/hyperparameters/'

path = bitbucket + 'Networks/' + str(task_name)
data_path = path + '/network_training_validation_losses.npz'
img_path = path + '/loss_curve.png'

# Load data from the NPZ file
data = np.load(data_path)

# Extract the relevant arrays
training_losses = data['training_losses']
validation_losses = data['validation_losses']
epochs = data['epochs']

# Plot the graph
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses vs. Epochs')
plt.legend()

# Crop graph to max validation_loss * 3
max_validation_loss = max(validation_losses)
plt.ylim(0, max_validation_loss * 3)

# Save the plot as an image file
plt.savefig(img_path)

# Display the plot
plt.show()
