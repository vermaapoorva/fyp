import numpy as np
import torch
from torch.utils.data import Dataset
from npy_append_array import NpyAppendArray
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class ImageToPoseDatasetCoarse(Dataset):

    def __init__(self, scene_name, amount_of_data):

        # self.dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'
        self.dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data/{scene_name}_data/'
        # Images are stored as image_0.npy, image_1.npy, etc.
        # Actions are stored as action_0.npy, action_1.npy, etc.
 

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, index):
        print("getting index: ", index)

        image = np.copy(self.images[index])
        if 1:
            noise = 0.1
            b_rand = np.random.uniform(-noise, noise)
            g_rand = np.random.uniform(-noise, noise)
            r_rand = np.random.uniform(-noise, noise)
            image[0] += np.tile(b_rand, image.shape[1:])
            image[1] += np.tile(g_rand, image.shape[1:])
            image[2] += np.tile(r_rand, image.shape[1:])
            image = image.clip(0, 1)

        image = image / 255
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        print(image.shape)

        # Save image as png locally
        if index%10000 == 0:
            saving_image = np.transpose(image, (1, 2, 0))
            saving_image = saving_image * 255
            saving_image = saving_image.astype(np.uint8)
            plt.imsave(f"images/image_{index}.png", saving_image)  

        action = np.copy(self.actions[index])
        endpoint_height = np.copy(self.endpoint_heights[index])
        example = {'image': image, 'action': action, 'endpoint_height': endpoint_height}
        
        return example
