import numpy as np
import torch
import torchvision
import cv2
import torch.nn.functional as F

# from Common import config
# from Common import utils

import os


class Network(torch.nn.Module):

    def __init__(self, network_path):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Set the class variables from the arguments
        self.network_path = network_path

    def delete(self):
        os.remove(self.network_path)

    def save(self, checkpoint_path=None):
        if checkpoint_path is None:
            torch.save(self.state_dict(), self.network_path)
        else:
            torch.save(self.state_dict(), checkpoint_path)

    def load(self):
        # If the network file doesn't exist (it has not been trained yet), then create a new one.
        if not os.path.exists(self.network_path):
            print('Creating network at: ' + str(self.network_path))
            self.save()

        # Otherwise (it has already been trained), use the existing file.
        else:
            print('Loading network from: ' + str(self.network_path))
            state_dict = torch.load(self.network_path)
            self.load_state_dict(state_dict)

    def load_from_checkpoint(self, checkpoint_path):
        print('Loading network from: ' + str(checkpoint_path))
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)

    def set_eval_dropout(self):
        self.apply(self.apply_dropout)

    @staticmethod
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    def freeze_features(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def get_num_trainable_params(self):
        # Each param in the below loop is one part of one layer
        # e.g. the first param is all the CNN weights in the first layer, the second param is all the biases in the first layer, the third param is all the CNN weights in the second layer, etc.
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params

    def print_architecture(self):
        print('Network architecture:')
        print(self)

    def print_weights(self):
        print('Network weights:')
        print('Feature extractor:')
        for name, param in self.feature_extractor.named_parameters():
            if 'weight' in name:
                print('name = ' + str(name))
                print('shape = ' + str(param.shape))
                print('values = ' + str(param[0, 0, 0]))
        print('Predictor:')
        for name, param in self.predictor.named_parameters():
            if 'weight' in name:
                print('name = ' + str(name))
                print('shape = ' + str(param.shape))
                print('values = ' + str(param[0, 0]))

    def print_gradients(self):
        print('Network gradients:')
        print('CNNs')
        for name, param in self.cnns[1].named_parameters():
            if 'weight' in name:
                print(param.grad[0, 0, 0, :3])
        print('FCs')
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                print(param.grad[0, :3])


# Image size: 64
class ImageToPoseNetworkCoarse(Network):

    def __init__(self, task_name, hyperparameters):
        # If the network directory doesn't exist (it has not been trained yet), then create a new one.
        # Otherwise (it has already been trained), use the existing directory.
        if not os.path.exists('/vol/bitbucket/av1019/dagger/hyperparameters/Networks/' + str(task_name)):
            os.makedirs('/vol/bitbucket/av1019/dagger/hyperparameters/Networks/' + str(task_name))
        # Call the parent constructor, which will set the save path.
        image_to_pose_network_path = '/vol/bitbucket/av1019/dagger/hyperparameters/Networks/' + str(task_name) + '/network.torch'
        Network.__init__(self, image_to_pose_network_path)
        # Define the network layers

        net_arch = [3] + hyperparameters['net_arch']
        layers = []

        for i in range(len(net_arch) - 1):
            layers.append(torch.nn.Conv2d(in_channels=net_arch[i], out_channels=net_arch[i + 1], kernel_size=3, stride=1, padding=0))
            layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0))
            layers.append(torch.nn.ReLU(inplace=False))

        self.conv = torch.nn.Sequential(*layers)

        input_shape = (3, 64, 64)
        # Iterate over the convolutional layers
        for layer in self.conv:
            if isinstance(layer, torch.nn.Conv2d):
                input_shape = (
                    layer.out_channels,
                    (input_shape[1] - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1,
                    (input_shape[2] - layer.kernel_size[1] + 2 * layer.padding[1]) // layer.stride[1] + 1,
                )
            elif isinstance(layer, torch.nn.MaxPool2d):
                input_shape = (
                    input_shape[0],
                    (input_shape[1] - layer.kernel_size[0]) // layer.stride + 1,
                    (input_shape[2] - layer.kernel_size[1]) // layer.stride + 1,
                )

        # Calculate the flat size
        self.flat_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.flat_size, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout(0.2),
        )
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(201, 50),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(50, 4)
        )

    def forward(self, input_image, height):
        # Compute the cnn features
        input_image = input_image.to(torch.float32)
        height = height.to(torch.float32)
        image_features = self.conv(input_image)
        image_features_flat = torch.reshape(image_features, (input_image.shape[0], -1))
        # Compute the mlp features
        mlp_features = self.mlp(image_features_flat)
        # Concatenate the z value
        combined_features = torch.cat((mlp_features, height), dim=1)
        # Make the prediction
        prediction = self.predictor(combined_features)
        return prediction