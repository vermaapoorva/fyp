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
# /vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_no_mp/pitcher_scene_data/
scene_name = 'cutlery_block_scene'
# root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/'
# shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/shards/'
# dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/{scene_name}/'

root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/'
shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_shards_initial_data/shards/'
dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'
# tar_file = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}.tar'

# Create 20 random shuffled mutually exclusive subsets of indicies from 0 to 10M
# indices = np.arange(10000)
# np.random.shuffle(indices)
# indices = np.array_split(indices, 2)
# print("len indicies: ", len(indices))
# # print all indices
# print(indices[0])
# print(indices[1])
# print("len indices[0]", len(indices[0]))
# print("len indices[1]", len(indices[1]))

# for i in range(len(indices)):
#     # np.save(f"indices_{i}.npy", indices[i])
#     print(f"indices_{i}", indices[i])
#     print("=======================================================================")

# Create 20 shards of data with indicies
# index = 1
# current_shard_indices = indices[index]
# amount_of_data = len(current_shard_indices)

amount_of_data=1000000
indices = np.arange(amount_of_data)
run_index = 0
start_index = amount_of_data*run_index
np.random.shuffle(indices)
count = 0
sink = wds.ShardWriter(f"{shards_dir}{scene_name}_train_shards-%06d.tar", maxcount=10000)
for i in indices:
    print(f"{count}/{amount_of_data} --- {(count/amount_of_data)*100}%")

    png_data = np.load(f"{dataset_directory}image_{start_index+i}.npy")
    npy_data = np.load(f"{dataset_directory}action_{start_index+i}.npy")

    sink.write(
        {"__key__": f"example{i:08d}",
        "input.npy": png_data,
        "output.npy": npy_data}
    )
    count += 1

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
