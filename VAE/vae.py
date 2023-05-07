import keras
from keras import layers

input_img = keras.Input(shape=(64, 64, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

print(encoded.shape)
print(decoded.shape)

autoencoder = keras.Model(input_img, decoded)
# autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mse')

from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split

x_train = np.load('x_train.npy') # input images
y_train = np.load('y_train.npy') # output 3x1 column matrices

x_train = x_train.astype('float32') / 255.

x_train, x_test, _, _ = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

autoencoder.save_weights("model.h5")
print("Saved model")

#################################################################################################
#################################################################################################

import matplotlib.pyplot as plt

# Load the model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
autoencoder = model_from_json(loaded_model_json)
autoencoder.load_weights("model.h5")
print("Loaded model")

# Create a new model with the encoder part of the autoencoder
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('max_pooling2d_2').output)

# Encode the test images to obtain the latent vectors
latent_vectors = encoder.predict(x_test)
print('Latent vector shape:', latent_vectors.shape)

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('output_50-epochs.png')
plt.show()