import torch
from torch.utils.data import IterableDataset

from torchvision import transforms
import webdataset as wds
import numpy as np
from itertools import islice
import time
import cv2
import os
import json

# set np random seed
np.random.seed(20)

import multiprocessing as mp
import numpy as np
import webdataset as wds

def write_data_to_tar(indices, shard_file_name, dataset_directory, start_index):
    print(f"Writing to {shard_file_name} at start index {start_index} with indices: {indices}...")
    sink = wds.TarWriter(shard_file_name)
    count = 0
    for i in indices:
        png_file = f"{dataset_directory}image_{start_index+i}.npy"
        action_file = f"{dataset_directory}action_{start_index+i}.npy"

        # If either file doesn't exist skip
        if not os.path.exists(png_file) or not os.path.exists(action_file):
            continue

        png_data = np.load(png_file)
        npy_data = np.load(action_file)

        sink.write(
            {"__key__": f"example{i:08d}",
            "input.npy": png_data,
            "output.npy": npy_data}
        )
        count += 1
    sink.close()
    print(f"Shard {shard_file_name} has {count} images")

    return {"shard file name": shard_file_name, "count": count}

amount_of_data = 500000
indices = np.arange(amount_of_data)
np.random.shuffle(indices)
shard_size = 10000
num_of_shards = amount_of_data // shard_size
run_index = 19
start_index = amount_of_data*run_index
current_shard_index = num_of_shards * run_index
start_indices = [start_index + i*shard_size for i in range(num_of_shards)]

print(f"start indices: {start_indices}, num of shards: {num_of_shards}")

scene_name = 'purple_block_scene_large_translation_noise'

root_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/'
shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/final_shards/'
dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'

# Create all directories if they don't exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(shards_dir):
    os.makedirs(shards_dir)

pool = mp.Pool(processes=mp.cpu_count())  # Use multiprocessing pool

shard_filenames = [f"{shards_dir}{scene_name}_shards-{i:06d}.tar" for i in range(current_shard_index, current_shard_index+num_of_shards)]

print(f"shard file names: {shard_filenames}")

print("Number of processes running in the pool:", pool._processes)

results = pool.starmap(write_data_to_tar, zip(np.array_split(indices, num_of_shards), shard_filenames, [dataset_directory]*num_of_shards, start_indices))

pool.close()
pool.join()

print(results)

with open(f"/vol/bitbucket/av1019/behavioural-cloning/c2f/purple_block_shards_{run_index}.json", "w") as f:
    json.dump(results, f, indent=4)





# def write_to_tar_file(indices, shard_file_name, dataset_directory):
#     print(f"Writing to {shard_file_name}...")
#     raw_dataset_directory = dataset_directory + f"raw_data_{iteration}/"
#     sink = wds.TarWriter(shard_file_name)
#     for count, i in enumerate(indices):

#         png_data = np.load(f"{raw_dataset_directory}image_{i}.npy")
#         npy_data = np.load(f"{raw_dataset_directory}action_{i}.npy")
#         sink.write(
#             {"__key__": f"example_{iteration}_{i:08d}",
#             "input.npy": png_data,
#             "output.npy": npy_data
#             })

# amount_of_data=500000
# indices = np.arange(amount_of_data)
# run_index = 19
# start_index = amount_of_data*run_index
# np.random.shuffle(indices)
# num_of_shards_per_run = 50
# current_index = num_of_shards_per_run*run_index

# print(f"len indices: {len(indices)} \n start_index: {start_index} \n amount_of_data: {amount_of_data} \n current_index: {current_index}")

# sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
# print(f"================New shard created index: {current_index}================")
# current_index += 1

# length_of_shard = 0
# for count, i in enumerate(indices):

#     if count % 10000 == 0 and count != 0:
#         print(f"Length of completed shard: {length_of_shard}")
#         sink = wds.TarWriter(f"{shards_dir}{scene_name}_shards-{current_index:06d}.tar")
#         print(f"================New shard created index: {current_index}================")
#         current_index += 1
#         length_of_shard = 0

#     png_data = np.load(f"{dataset_directory}image_{start_index+i}.npy")
#     npy_data = np.load(f"{dataset_directory}action_{start_index+i}.npy")

#     sink.write(
#         {"__key__": f"example{i:08d}",
#         "input.npy": png_data,
#         "output.npy": npy_data}
#     )
#     length_of_shard += 1

# print(f"Length of completed shard: {length_of_shard}")



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
