from npy_append_array import NpyAppendArray
import numpy as np

# Define the file names and the final file name
# file_names = ["cutlery_block_scene_0_0_actions.npy", "cutlery_block_scene_0_1_actions.npy",
#               "cutlery_block_scene_0_2_actions.npy", ...]  # Add all the file names
# final_file_name = "cutlery_block_scene_0_actions.npy"
# /vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/cutlery_block_scene_0/cutlery_block_scene_0_0_actions.npy

scene_name = "cutlery_block_scene"
bottleneck_index = 0
content = "heights"

root_dir = f"/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}_{bottleneck_index}/"

final_file_name = f"{root_dir}{scene_name}_{bottleneck_index}_{content}.npy"
# final_array = NpyAppendArray(final_file_name, mode='mmap')

print(f"Creating final file: {final_file_name}")

with NpyAppendArray(final_file_name) as final_file:
    for i in range(0, 10000000, 1000000):
        file_name = f"{root_dir}{scene_name}_{bottleneck_index}_{i}_{content}.npy"

        # If file exists append it to the final file
        try:
            print(f"Appending file: {file_name}")
            file = np.load(file_name, mmap_mode='r')
            final_file.append(file)
        except FileNotFoundError:
            print(f"File {file_name} not found")
            continue
    
    print(f"Final file: {final_file_name} created")
    # Print length of final file
    # print(f"Length of final file: {len(final_file)}")