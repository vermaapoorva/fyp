import os

shards_dir = '/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/final_shards/'
scene_name = "bowl_scene"
# Iterate through the files in the directory
count = 550
for index, filename in enumerate(os.listdir(shards_dir)):
    if filename.startswith(f"{scene_name}_large_translation_noise_shards-"):
        new_filename = f"{scene_name}_large_translation_noise_2_shards-{count:06d}.tar"
        os.rename(os.path.join(shards_dir, filename), os.path.join(shards_dir, new_filename))
        count += 1
    # if count == 550:
    #     break

print(count)