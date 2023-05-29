import torch
from torch.utils.data import IterableDataset

from torchvision import transforms
import webdataset as wds
import numpy as np
from itertools import islice
import time
import cv2

# set np random seed
np.random.seed(2023)

scene_name = 'bowl_scene'
root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/'
shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/shards/'
dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'
# tar_file = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}.tar'

# Create 20 random shuffled mutually exclusive subsets of indicies from 0 to 10M
indices = np.arange(10000000)
np.random.shuffle(indices)
indices = np.array_split(indices, 20)
# for i in range(len(indices)):
#     # np.save(f"indices_{i}.npy", indices[i])
#     print(f"indices_{i}", indices[i])
#     print("=======================================================================")

# Create 20 shards of data with indicies
index = 19
current_shard_indices = indices[index]
amount_of_data = len(current_shard_indices)

sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{index:06d}.tar")
for i in range(amount_of_data):
    print(f"{i}/{amount_of_data} --- {(i/amount_of_data)*100}%")

    png_data = cv2.imread(f"{dataset_directory}image_{current_shard_indices[i]}.png")
    npy_data = np.load(f"{dataset_directory}image_{current_shard_indices[i]}.npy")

    sink.write(
        {"__key__": f"example{current_shard_indices[i]:08d}",
        "png": png_data,
        "npy": npy_data}
    )

# amount_of_data = 10000
# # shard dataset using wds.shardwriter
# sink = wds.ShardWriter(f"{root_dir}cutlery_block_shards-%06d.tar", maxcount=1000)
# for i in range(10000):
#     print(f"{i}/{amount_of_data} --- {(i/amount_of_data)*100}%")

#     png_data = open(f"{dataset_directory}image_{i}.png", "rb").read()

#     npy_data = np.load(f"{dataset_directory}image_{i}.npy")

#     sink.write(
#         {"__key__": f"example{i:08d}",
#         "png": png_data,
#         "npy": npy_data}
#     )


# sharded_dataset_directory = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/cutlery_block_shards-{000000..000009}.tar'

# dataset = (wds.WebDataset(sharded_dataset_directory).decode("torchrgb").to_tuple("png", "npy"))

# for sample in islice(dataset, 0, 3):
#     print("sample:", sample)
#     # for key, value in sample.items():
#     #     print("key:", key)
#     #     print("value:", value)


# training_dataset_directory = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/cutlery_block_shards-{000000..000008}.tar'
# validation_dataset_directory = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/cutlery_block_shards-{000009..000009}.tar'


# start = time.time()
# image_to_pose_training_dataset = wds.WebDataset(training_dataset_directory).shuffle(100)

# # .shuffle(1000)
# # .decode("torchrgb").to_tuple("png", "npy")
# image_to_pose_validation_dataset = wds.WebDataset(validation_dataset_directory).shuffle(100)
# # .decode("torchrgb").to_tuple("png", "npy")

# # training_loader = wds.WebLoader(image_to_pose_training_dataset.batched(64), batch_size=None, shuffle=False)
# # validation_loader = wds.WebLoader(image_to_pose_validation_dataset.batched(64), batch_size=None, shuffle=False)

# for sample in islice(image_to_pose_training_dataset, 0, 3):
#     # print("sample:", sample)
#     for key, value in sample.items():
#         # print("key:", key)
#         if key == '__key__':
#             print("value:", value)
#         # print("value:", value)

# end = time.time()
# print("time taken 100:", end-start)