import torch
from torch.utils.data import IterableDataset

from torchvision import transforms
import webdataset as wds
from itertools import islice

scene_name = 'cutlery_block_scene'
dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}.tar'

# dataset = wds.WebDataset(dataset_directory).shuffle(1000).decode("rgb").to_tuple("png", "npy").map(preprocess)
# dataset = dataset.compose(get_patches)
# dataset = dataset.batched(16)

# loader = wds.WebLoader(dataset, num_workers=4, batch_size=None)
# loader = loader.unbatched().shuffle(1000).batched(12)

# batch = next(iter(loader))
# print(batch[0].shape, batch[1].shape)

print("creating dataset")
dataset = wds.WebDataset(dataset_directory).shuffle(1000).decode("torchrgb").to_tuple("png", "npy")
print("created dataset")

print("creating dataloader")
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)
print("created dataloader")

for inputs, outputs in dataloader:
    # print(inputs.shape, outputs.shape)
    # Save image
    image = transforms.ToPILImage()(inputs[0])
    image.save(f'test2.png')

# i=0
# for sample in islice(dataset, 0, 3):
#     for key, value in sample.items():
#         print("key:", key)
#         print("value:", value)

#         if key == 'png':
#             # Save image
#             image = transforms.ToPILImage()(value)
#             image.save(f'test_{i}.png')
#             i += 1
#         # print(key, repr(value)[:50])
#     print()
