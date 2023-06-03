import os

shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/shards/'

# Iterate through the files in the directory
count = 0
for index, filename in enumerate(os.listdir(shards_dir)):
    if filename.startswith("wooden_block_scene"):
        new_filename = f"wooden_block_scene_large_translation_noise_shards-{count:06d}.tar"
        os.rename(os.path.join(shards_dir, filename), os.path.join(shards_dir, new_filename))
        count += 1

print(count)