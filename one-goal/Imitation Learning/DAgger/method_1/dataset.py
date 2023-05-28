import numpy as np
import torch
from torch.utils.data import Dataset
from npy_append_array import NpyAppendArray
import cv2
import os

class ImageToPoseDatasetCoarse(Dataset):

    def __init__(self, scene_name, task_name):

        self.scene_name = scene_name
        self.dataset_directory = f'/vol/bitbucket/av1019/dagger/final_expert_data_npy/{scene_name}/'

        # Make directory if it doesn't exist
        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)
        
        self.dataset_directory += f'{task_name}/'

        # Make directory if it doesn't exist
        if not os.path.exists(self.dataset_directory):
            os.makedirs(self.dataset_directory)

        self.action_file = f'{self.dataset_directory}{scene_name}_actions.npy'
        self.height_file = f'{self.dataset_directory}{scene_name}_heights.npy'
        self.image0_file = f'{self.dataset_directory}{scene_name}_images_0.npy'
        self.image1_file = f'{self.dataset_directory}{scene_name}_images_1.npy'
        self.image2_file = f'{self.dataset_directory}{scene_name}_images_2.npy'
        self.image3_file = f'{self.dataset_directory}{scene_name}_images_3.npy'
        self.image4_file = f'{self.dataset_directory}{scene_name}_images_4.npy'
        self.image5_file = f'{self.dataset_directory}{scene_name}_images_5.npy'
        self.image6_file = f'{self.dataset_directory}{scene_name}_images_6.npy'
        self.image7_file = f'{self.dataset_directory}{scene_name}_images_7.npy'
        self.image8_file = f'{self.dataset_directory}{scene_name}_images_8.npy'
        self.image9_file = f'{self.dataset_directory}{scene_name}_images_9.npy'
        self.image10_file = f'{self.dataset_directory}{scene_name}_images_10.npy'
        self.image11_file = f'{self.dataset_directory}{scene_name}_images_11.npy'
        self.image12_file = f'{self.dataset_directory}{scene_name}_images_12.npy'
        self.image13_file = f'{self.dataset_directory}{scene_name}_images_13.npy'
        self.image14_file = f'{self.dataset_directory}{scene_name}_images_14.npy'
        self.image15_file = f'{self.dataset_directory}{scene_name}_images_15.npy'
        self.image16_file = f'{self.dataset_directory}{scene_name}_images_16.npy'
        self.image17_file = f'{self.dataset_directory}{scene_name}_images_17.npy'
        self.image18_file = f'{self.dataset_directory}{scene_name}_images_18.npy'
        self.image19_file = f'{self.dataset_directory}{scene_name}_images_19.npy'
        self.image20_file = f'{self.dataset_directory}{scene_name}_images_20.npy'

        self.actions = np.empty((0, 4))
        self.endpoint_heights = np.empty((0, 1))
        self.images0 = np.empty((0, 3, 64, 64))
        self.images1 = np.empty((0, 3, 64, 64))
        self.images2 = np.empty((0, 3, 64, 64))
        self.images3 = np.empty((0, 3, 64, 64))
        self.images4 = np.empty((0, 3, 64, 64))
        self.images5 = np.empty((0, 3, 64, 64))
        self.images6 = np.empty((0, 3, 64, 64))
        self.images7 = np.empty((0, 3, 64, 64))
        self.images8 = np.empty((0, 3, 64, 64))
        self.images9 = np.empty((0, 3, 64, 64))
        self.images10 = np.empty((0, 3, 64, 64))
        self.images11 = np.empty((0, 3, 64, 64))
        self.images12 = np.empty((0, 3, 64, 64))
        self.images13 = np.empty((0, 3, 64, 64))
        self.images14 = np.empty((0, 3, 64, 64))
        self.images15 = np.empty((0, 3, 64, 64))
        self.images16 = np.empty((0, 3, 64, 64))
        self.images17 = np.empty((0, 3, 64, 64))
        self.images18 = np.empty((0, 3, 64, 64))
        self.images19 = np.empty((0, 3, 64, 64))
        self.images20 = np.empty((0, 3, 64, 64))

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, index):
        image = np.copy(self.images[index])
        action = np.copy(self.actions[index])
        endpoint_height = np.copy(self.endpoint_heights[index])
        example = {'image': image, 'action': action, 'endpoint_height': endpoint_height}
        return example

    def update(self, iteration):
        # if action file exists and has data, load it
        if os.path.isfile(self.action_file) and os.path.getsize(self.action_file) > 0:
            self.actions = np.load(self.action_file, mmap_mode='r').astype(np.float32)
            print(f"Loaded {self.actions.shape[0]} actions")

        # if height file exists and has data, load it
        if os.path.isfile(self.height_file) and os.path.getsize(self.height_file) > 0:
            self.endpoint_heights = np.load(self.height_file, mmap_mode='r').astype(np.float32)
            print(f"Loaded {self.endpoint_heights.shape[0]} heights")

        amount_of_data = self.actions.shape[0]

        # Load images in self.images{iteration}_file
        

        # if image file exists and has data, load it
        # if os.path.isfile(self.image_file) and os.path.getsize(self.image_file) > 0:
        #     self.images = np.memmap(f'{self.dataset_directory}{self.scene_name}_images.npy', dtype=np.float32, mode='r', shape=(amount_of_data, 3, 64, 64))
        #     print(f"Loaded {self.images.shape[0]} images")