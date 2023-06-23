# get files image_{x}.npy from /home/apoorva/ and save png in traj_images/ folder

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Make traj_images dir
if not os.path.exists('traj_images'):
    os.makedirs('traj_images')

# Get the files
for i in range(2):
    file = f'/home/apoorva/image_{i}.npy'
    img = np.load(file)
    # channel first to channel last
    img = np.moveaxis(img, 0, -1)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'traj_images/image_{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close()