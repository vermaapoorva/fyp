import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_images_as_png(scene_name):
    dataset_directory = f'/vol/bitbucket/av1019/behavioural-cloning/c2f/final_expert_data_npy/{scene_name}/'
    output_filename = f'{scene_name}_0_images.npy'

    # Load images from the npy file using memmap
    images = np.memmap(os.path.join(dataset_directory, output_filename), dtype=np.float32, mode='r', shape=(1001, 64, 64, 3))
    num_images = images.shape[0]

    # min max of images
    print("min:", np.min(images))
    print("max:", np.max(images))

    print(f'Loaded {num_images} images from {output_filename}')

    # Create a directory to save the PNG images
    output_directory = f'{scene_name}_png'
    os.makedirs(output_directory, exist_ok=True)

    # Save each image as a PNG file
    for i in range(num_images):
        if np.min(image) < 0 or np.max(image) > 1:
            print("ERROR: image values out of range")
            print("min:", np.min(image))
            print("max:", np.max(image))
        if i%100 == 0:
            image = np.copy(images[i])
            image_filename = f'image_{i}.png'
            print(image)
            # print min max in image

            # print("image shape before:", image.shape)
            # image = np.transpose(image, (2, 0, 1))
            # print("image shape after:", image.shape)
            # image *= 255
            # image = (image * 255).astype(np.uint8)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR color channel order
            image_path = os.path.join(output_directory, image_filename)

            plt.imsave(image_path, image)
            print(f'Saved image {i} as {image_path}')

    print('All images saved as PNG.')

# Usage example
scene_name = 'cutlery_block_scene2'
save_images_as_png(scene_name)
