import os
import cv2
from npy_append_array import NpyAppendArray
import numpy as np

scene_name = "cutlery_block_scene"
bottleneck_index = 0

root_dir = f"/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}_{bottleneck_index}/"
image_directory = f"{root_dir}images/"

start_index = 0
end_index = 300000

# Initialize NpyAppendArray with the desired npy file path and shape
final_file_name = f"{root_dir}{scene_name}_{bottleneck_index}_images.npy"

with NpyAppendArray(final_file_name, (end_index - start_index, 64, 64, 3)) as final_file:
    for i in range(start_index, end_index):
        image_path = os.path.join(image_directory, f'image_{i}.png')
        image = cv2.imread(image_path)
        image = np.expand_dims(image, axis=0)
        final_file.append(image)
        if i % 1000 == 0:
            print(f"Appended {i}/{end_index} --- {i / end_index * 100}%")

print("Images saved to npy file:", npy_file)
