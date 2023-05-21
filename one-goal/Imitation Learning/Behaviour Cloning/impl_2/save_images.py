import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.loads(f.read())
    return data

train_data = load_data("/vol/bitbucket/av1019/behavioural-cloning/hyperparameters/expert_data/expert_data_white_bead_mug_10000.pkl")
observations = np.array(train_data['observations'])

for i, image in enumerate(observations):
    # make it channel last
    image = np.transpose(image, (1, 2, 0))
    # save image
    image_file_name = "images/image_" + str(i) + ".png"
    plt.imsave(image_file_name, image)