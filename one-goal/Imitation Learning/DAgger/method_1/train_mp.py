import numpy as np
import cv2
import torch.cuda
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, BatchSampler
import matplotlib.pyplot as plt
import math

from dataset import ImageToPoseDatasetCoarse

# from Common import utils
# from Common import config
# from Common import graphs
from network import ImageToPoseNetworkCoarse
import webdataset as wds

import multiprocessing as mp

import os
import pickle

def write_to_tar_file(indices, shard_file_name, iteration, dataset_directory):
    print(f"Writing to {shard_file_name}...")
    raw_dataset_directory = dataset_directory + f"raw_data_{iteration}/"
    sink = wds.TarWriter(shard_file_name)
    for count, i in enumerate(indices):

        png_data = np.load(f"{raw_dataset_directory}image_{i}.npy")
        npy_data = np.load(f"{raw_dataset_directory}action_{i}.npy")
        sink.write(
            {"__key__": f"example_{iteration}_{i:08d}",
            "input.npy": png_data,
            "output.npy": npy_data
            })

class ImageToPoseTrainerCoarse:

    def __init__(self, task_name, scene_name, hyperparameters):

        self.bitbucket = '/vol/bitbucket/av1019/dagger/hyperparameters/'
        self.task_name = task_name
        self.hyperparameters = hyperparameters

        self.dataset_directory = f'/vol/bitbucket/av1019/dagger/hyperparameters/data/{task_name}/'

        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)

        self.image_to_pose_network = ImageToPoseNetworkCoarse(task_name, hyperparameters)

        self.minibatch_size = hyperparameters['batch_size']
        self.init_learning_rate = hyperparameters['learning_rate']
        self.loss_orientation_coefficient = 0.01

        # INITIALISE THE NETWORK
        # Set the GPU
        torch.cuda.set_device('cuda:0')
        self.image_to_pose_network.cuda()
        # Define the optimiser
        self.loss_function = torch.nn.MSELoss(reduction='none')
        self.optimiser = torch.optim.Adam(self.image_to_pose_network.parameters(), lr=self.init_learning_rate)
        self.lr_patience = 10
        self.patience = 15
        # self.lr_patience = 20
        # self.patience = 20

    def get_dataset_dir(self):
        return self.dataset_directory

    def get_network(self):
        return self.image_to_pose_network

    def update_dataloaders(self, iteration, amount_of_training_data, amount_of_validation_data, training_shards_next_index, validation_shards_next_index):

        print("Updating dataloaders...")
        print(f"iteration: {iteration}, amount_of_training_data: {amount_of_training_data}, amount_of_validation_data: {amount_of_validation_data}, training_shards_next_index: {training_shards_next_index}, validation_shards_next_index: {validation_shards_next_index}")
        raw_dataset_directory = self.dataset_directory + f"raw_data_{iteration}/"

        indices = np.arange(amount_of_training_data + amount_of_validation_data)
        np.random.shuffle(indices)
        training_indices = indices[:amount_of_training_data]
        validation_indices = indices[amount_of_training_data:]

        training_shard_indices = np.array_split(training_indices, math.ceil(len(training_indices) / 10000))
        validation_shard_indices = np.array_split(validation_indices, math.ceil(len(validation_indices) / 10000))

        print("Length of training shard indices: ", len(training_shard_indices))
        print("Length of validation shard indices: ", len(validation_shard_indices))

        pool = mp.Pool(processes=mp.cpu_count())  # Use multiprocessing pool
        training_shard_filenames = [f"{self.dataset_directory}training_shards-{training_shards_next_index + i:06d}.tar" for i in range(len(training_shard_indices))]
        validation_shard_filenames = [f"{self.dataset_directory}validation_shards-{validation_shards_next_index + i:06d}.tar" for i in range(len(validation_shard_indices))]

        print("Number of processes running in the pool:", pool._processes)

        # print("Training shard filenames: ", training_shard_filenames)
        # print("Validation shard filenames: ", validation_shard_filenames)

        print("Creating new .tar files for the training dataset...")
        pool.starmap(write_to_tar_file, zip(training_shard_indices, training_shard_filenames, [iteration] * len(training_shard_indices), [self.dataset_directory] * len(training_shard_indices)))

        print("Creating new .tar files for the validation dataset...")
        pool.starmap(write_to_tar_file, zip(validation_shard_indices, validation_shard_filenames, [iteration] * len(validation_shard_indices), [self.dataset_directory] * len(validation_shard_indices)))

        pool.close()
        pool.join()

        training_shards_next_index += len(training_shard_indices)
        validation_shards_next_index += len(validation_shard_indices)

        print("Finished creating new .tar files for the training and validation datasets...")

        self.training_dataset_directory = self.dataset_directory + f'training_shards-{{000000..{training_shards_next_index-1:06d}}}.tar'
        self.validation_dataset_directory = self.dataset_directory + f'validation_shards-{{000000..{validation_shards_next_index-1:06d}}}.tar'

        self.image_to_pose_training_dataset = wds.WebDataset(self.training_dataset_directory).decode().to_tuple("input.npy", "output.npy")
        self.image_to_pose_validation_dataset = wds.WebDataset(self.validation_dataset_directory).decode().to_tuple("input.npy", "output.npy")

        self.training_loader = DataLoader(self.image_to_pose_training_dataset, batch_size=self.minibatch_size, num_workers=8, pin_memory=True)
        self.validation_loader = DataLoader(self.image_to_pose_validation_dataset, batch_size=self.minibatch_size, num_workers=8, pin_memory=True)

        return training_shards_next_index, validation_shards_next_index

    def reset_network(self):
        self.image_to_pose_network.delete()
        self.image_to_pose_network = ImageToPoseNetworkCoarse(self.task_name, self.hyperparameters)
        torch.cuda.set_device('cuda:0')
        self.image_to_pose_network.cuda()
        # Define the optimiser
        self.optimiser = torch.optim.Adam(self.image_to_pose_network.parameters(), lr=self.init_learning_rate)
        print("Reset network...")

    def train(self, iteration):
        print('Training the network...')
        # Loop over epochs
        training_losses = []
        validation_losses = []
        validation_errors = []
        min_validation_loss = np.inf
        num_bad_epochs_since_lr_change = 0
        num_bad_epochs = 0
        epoch_num = 0
        while True:
            print('Epoch: ' + str(epoch_num))
            # Increment the epoch num
            epoch_num += 1
            # TRAINING
            print('Training...')
            # Set to training mode
            self.image_to_pose_network.train()
            # Set some variables to store the training results
            training_epoch_loss_sum = 0
            # Loop over minibatches
            num_minibatches = 0
            for minibatch_num, examples in enumerate(self.training_loader):
            # for examples in self.training_loader:
                # print(f'minibatch {minibatch_num}/{len(self.training_loader)}')
                # Do a forward pass on this minibatch
                minibatch_loss = self._train_on_minibatch(examples, epoch_num, num_minibatches)
                # Update the loss sums
                training_epoch_loss_sum += minibatch_loss
                # Update the number of minibatches processed
                num_minibatches += 1
            # Store the training losses
            training_loss = training_epoch_loss_sum / num_minibatches
            training_losses.append(training_loss)

            # VALIDATION
            print('Validating...')
            # Set to validation mode
            self.image_to_pose_network.eval()
            # Set some variables to store the training results
            validation_epoch_loss_sum = 0
            validation_epoch_x_error_sum = 0
            validation_epoch_y_error_sum = 0
            validation_epoch_z_error_sum = 0
            validation_epoch_theta_error_sum = 0
            if 0:
                xy_error_sum = np.zeros([10, 10], dtype=np.float32)
                xy_error_count = np.zeros([10, 10], dtype=np.uint8)
                bins = np.linspace(-0.04, 0.05, 10)  # The bin numbers represent the value on the right of the bin (i.e. the maximum value in that bin)
            # Loop over minibatches
            num_minibatches = 0
            for minibatch_num, examples in enumerate(self.validation_loader):
            # for examples in self.validation_loader:
                # print(f'minibatch {minibatch_num}/{len(self.validation_loader)}')
                # Do a forward pass on this minibatch
                minibatch_loss, minibatch_x_error, minibatch_y_error, minibatch_z_error, minibatch_theta_error, minibatch_poses = self._validate_on_minibatch(examples, epoch_num)
                # Update the loss sums
                validation_epoch_loss_sum += minibatch_loss
                validation_epoch_x_error_sum += minibatch_x_error
                validation_epoch_y_error_sum += minibatch_y_error
                validation_epoch_z_error_sum += minibatch_z_error
                validation_epoch_theta_error_sum += minibatch_theta_error
                # Update the errors for each position
                # if 0:
                #     minibatch_x_bins = np.digitize(minibatch_poses[:, 0], bins)
                #     minibatch_y_bins = np.digitize(minibatch_poses[:, 1], bins)
                #     xy_error_sum[minibatch_x_bins, minibatch_y_bins] += 0.5 * (minibatch_x_error + minibatch_y_error)
                #     xy_error_count[minibatch_x_bins, minibatch_y_bins] += 1
                # Update the number of minibatches processed
                num_minibatches += 1
            # Store the validation losses
            validation_loss = validation_epoch_loss_sum / num_minibatches
            validation_losses.append(validation_loss)
            validation_epoch_x_error = validation_epoch_x_error_sum / num_minibatches
            validation_epoch_y_error = validation_epoch_y_error_sum / num_minibatches
            validation_epoch_z_error = validation_epoch_z_error_sum / num_minibatches
            validation_epoch_theta_error = validation_epoch_theta_error_sum / num_minibatches
            validation_error = [validation_epoch_x_error, validation_epoch_y_error, validation_epoch_z_error, validation_epoch_theta_error]
            validation_errors.append(validation_error)

            # Decide whether to update the number of epochs that have elapsed since the loss decreased
            # A 'bad epoch' is one where the loss does not decrease by at least 1% of the current minimum loss
            if validation_loss > 0.99 * min_validation_loss:
                num_bad_epochs_since_lr_change += 1
                num_bad_epochs += 1
            else:
                num_bad_epochs_since_lr_change = 0
                num_bad_epochs = 0
            print('Epoch ' + str(epoch_num) + ': num bad epochs = ' + str(num_bad_epochs))
            # Decide whether to reduce the learning rate
            if num_bad_epochs_since_lr_change > self.lr_patience:
                for p in self.optimiser.param_groups:
                    old_lr = p['lr']
                    new_lr = 0.5 * old_lr
                    p['lr'] = new_lr
                print('Dropping learning rate to ' + str(new_lr))
                num_bad_epochs_since_lr_change = 0
            # Decide whether this loss is the minimum so far. If so, set the minimum loss and save the network.
            # This needs to come after updating the number of bad epochs, otherwise the comparison of the loss and min loss could be the same number
            if validation_loss < min_validation_loss:
                min_validation_loss = validation_loss
                min_validation_error = validation_error
                self.image_to_pose_network.save()
            # Decide whether to do early stopping
            if num_bad_epochs > self.patience:
                break

            window = 10
            mask = np.ones(window) / window
            if len(training_losses) >= window:
                running_average_training_losses = np.convolve(training_losses, mask, mode='valid')
                running_average_validation_losses = np.convolve(validation_losses, mask, mode='valid')
                running_average_epochs = np.arange(start=1 + 0.5 * (window - 1), stop=len(training_losses) - 0.5 * (window - 1) + 1, step=1)
                epochs = range(1, epoch_num + 1)

                # Save as dictionary - training losses, validation losses, epochs, running_average_training_losses, running_average_validation_losses, running_average_epochs and validation errors to one file
                np.savez(self.bitbucket + 'Networks/' + str(self.task_name) + '/network_training_validation_losses.npz', training_losses=training_losses, validation_losses=validation_losses, epochs=epochs, running_average_training_losses=running_average_training_losses, running_average_validation_losses=running_average_validation_losses, running_average_epochs=running_average_epochs, validation_errors=validation_errors)

                print('Epoch ' + str(epoch_num) + ':')
                print('\tRunning Average: Training loss: ' + str(running_average_training_losses[-1]) + ', Validation loss: ' + str(running_average_validation_losses[-1]))
                print('\tTraining loss: ' + str(training_loss) + ', Validation loss: ' + str(validation_loss))
                print('\tValidation position x error: ' + str(validation_epoch_x_error) + ', Validation position y error: ' + str(validation_epoch_y_error) + ', Validation orientation error: ' + str(validation_epoch_theta_error))

        # Save the error, so it can be used as a prior on uncertainty
        np.save(self.bitbucket + 'Networks/' + str(self.task_name) + '/network_uncertainty_validation_error.npy', min_validation_error)

        # Save a checkpoint
        checkpoint_path = self.bitbucket + 'Networks/' + str(self.task_name) + '/network_checkpoint_iteration_' + str(iteration) + '.torch'
        self.image_to_pose_network.save(checkpoint_path)

        # Return the minimum loss and error
        return min_validation_loss, min_validation_error

    def _train_on_minibatch(self, examples, epoch_num, num_minibatches=0):
        # Do a forward pass
        image_tensor = examples[0]
        # Create the z tensor, which needs to go from one dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        endpoint_height_tensor = torch.unsqueeze(examples[1][:, -1], 1).to(dtype=torch.float32)
        predictions = self.image_to_pose_network.forward(image_tensor.cuda(), endpoint_height_tensor.cuda())
        # Compute the loss
        ground_truths = examples[1][:, :-1].to(dtype=torch.float32).cuda()
        loss = self._compute_loss(predictions, ground_truths)
        # Set the gradients to zero
        self.optimiser.zero_grad()
        # Do a backward pass, which computes and stores the gradients
        loss.backward()
        # Do a weight update
        self.optimiser.step()

        # Return the loss
        minibatch_loss = loss.item()
        return minibatch_loss

    def _validate_on_minibatch(self, examples, epoch_num):        
        # Do a forward pass
        image_tensor = examples[0]
        # Create the z tensor, which needs to go from one dimension to two dimensions (batch dim, feature dim) in order for it to later be concatenated with the feature
        endpoint_height_tensor = torch.unsqueeze(examples[1][:, -1], 1).to(dtype=torch.float32)
        predictions = self.image_to_pose_network.forward(image_tensor.cuda(), endpoint_height_tensor.cuda())
        # Compute the loss
        ground_truths = examples[1][:, :-1].to(dtype=torch.float32).cuda()

        # Note that you need to call item() in the below, otherwise the loss will never be freed from cuda memory
        minibatch_loss = self._compute_loss(predictions, ground_truths).item()
        # Calculate the error
        minibatch_x_error, minibatch_y_error, minibatch_z_error, minibatch_theta_error = self._compute_errors(predictions.detach().cpu().numpy(), ground_truths.detach().cpu().numpy())
        # Get the x, y, z positions, so that we can plot the validation error at each position
        # minibatch_poses = examples[1][:, :-1].reshape(-1, 4).numpy()
        minibatch_poses = examples[1][:, :-1].numpy()
        return minibatch_loss, minibatch_x_error, minibatch_y_error, minibatch_z_error, minibatch_theta_error, minibatch_poses

    def _compute_loss(self, predictions, ground_truths):
        position_loss = self.loss_function(predictions[:, :3], ground_truths[:, :3]).mean()
        orientation_loss = self.loss_function(predictions[:, 3:], ground_truths[:, 3:]).mean()
        loss = position_loss + self.loss_orientation_coefficient * orientation_loss
        return loss

    def _compute_errors(self, predictions, ground_truths):
        x_errors = np.fabs(predictions[:, 0] - ground_truths[:, 0])
        x_error = x_errors.mean(axis=0)
        y_errors = np.fabs(predictions[:, 1] - ground_truths[:, 1])
        y_error = y_errors.mean(axis=0)
        z_errors = np.fabs(predictions[:, 2] - ground_truths[:, 2])
        z_error = z_errors.mean(axis=0)
        num_examples = len(predictions)
        theta_errors = np.zeros([num_examples], dtype=np.float32)
        for example_num in range(num_examples):
            theta_errors[example_num] = self.compute_absolute_angle_difference(predictions[example_num, 3], ground_truths[example_num, 3])
        theta_error = np.mean(theta_errors)
        return x_error, y_error, z_error, theta_error

    def compute_absolute_angle_difference(self, angle_1, angle_2):
        angle = np.pi - np.abs(np.abs(angle_1 - angle_2) - np.pi)
        return angle