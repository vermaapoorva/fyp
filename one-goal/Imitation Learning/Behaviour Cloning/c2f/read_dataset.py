import torch
from torch.utils.data import IterableDataset

from torchvision import transforms
import webdataset as wds
import numpy as np
from itertools import islice
import time
import cv2
import os

# set np random seed
np.random.seed(20)
# /vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_no_mp/pitcher_scene_data/
scene_name = 'teapot_scene_large_translation_noise'
# root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/'
# shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/shards/'
# dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/expert_data/{scene_name}/'

root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/'
shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/final_shards/'
dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'
# tar_file = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}.tar'

# Create all directories if they don't exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(shards_dir):
    os.makedirs(shards_dir)

amount_of_data=500000
indices = np.arange(amount_of_data)
run_index = 19
start_index = amount_of_data*run_index
np.random.shuffle(indices)
num_of_shards_per_run = 50
current_index = num_of_shards_per_run*run_index

print(f"len indices: {len(indices)} \n start_index: {start_index} \n amount_of_data: {amount_of_data} \n current_index: {current_index}")

sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
print(f"================New shard created index: {current_index}================")
current_index += 1

length_of_shard = 0
for count, i in enumerate(indices):

    if count % 10000 == 0 and count != 0:
        print(f"Length of completed shard: {length_of_shard}")
        sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
        print(f"================New shard created index: {current_index}================")
        current_index += 1
        length_of_shard = 0

    png_data = np.load(f"{dataset_directory}image_{start_index+i}.npy")
    npy_data = np.load(f"{dataset_directory}action_{start_index+i}.npy")

    sink.write(
        {"__key__": f"example{i:08d}",
        "input.npy": png_data,
        "output.npy": npy_data}
    )
    length_of_shard += 1

print(f"Length of completed shard: {length_of_shard}")



# # Skip if next shard exists and remove 10000 from start of indices
# while os.path.exists(f"{shards_dir}{scene_name}_shards-{current_index+1:06d}.tar"):
#     print(f"================Next shard exists index: {current_index+1}================")
#     start_index += 10000
#     current_index += 1

# # Create new shard
# sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
# print(f"================New shard created index: {current_index}================")
# current_index += 1
# print("starting index: ", start_index)

# length_of_shard = 0
# for count, i in enumerate(indices):

#     if (count >= start_index - (amount_of_data*run_index)):

#         if count % 10000 == 0 and count != 0 and count != start_index - (amount_of_data*run_index):
#             print(f"Length of completed shard: {length_of_shard}")
#             sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
#             print(f"================New shard created index: {current_index}================")
#             current_index += 1
#             length_of_shard = 0

#         png_data = np.load(f"{dataset_directory}image_{start_index+i}.npy")
#         npy_data = np.load(f"{dataset_directory}action_{start_index+i}.npy")

#         sink.write(
#             {"__key__": f"example{i:08d}",
#             "input.npy": png_data,
#             "output.npy": npy_data}
#         )
#         length_of_shard += 1

# print(f"Length of completed shard: {length_of_shard}")

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
