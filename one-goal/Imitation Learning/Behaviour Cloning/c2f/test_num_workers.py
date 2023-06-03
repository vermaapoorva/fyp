from time import time
import multiprocessing as mp
import webdataset as wds
from torch.utils.data import DataLoader

for pin_memory in [True, False]:
    for num_workers in range(2, 18, 2):  

        training_dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/shards/cutlery_block_scene_large_translation_noise_shards-{{000000..000099}}.tar'

        minibatch_size = 64

        image_to_pose_training_dataset = wds.WebDataset(training_dataset_directory).decode().to_tuple("input.npy", "output.npy")

        training_loader = DataLoader(image_to_pose_training_dataset, batch_size=minibatch_size, num_workers=num_workers, pin_memory=pin_memory)

        start = time()

        for epoch in range(1, 3):
            for i, data in enumerate(training_loader, 0):
                pass

        end = time()

        print("Finish with:{} second, num_workers={}, pin_memory={}".format(end - start, num_workers, pin_memory))