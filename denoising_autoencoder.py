from keras.datasets import mnist
import numpy as np
import keras.backend as K
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Model
from utils import plot_digits, plot_chart_accuracy, plot_chart_loss
from deep_autoencoder import create_deep_dense_ae

batch_size = 16

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test,  (len(x_test),  28, 28, 1))

encoder, decoder, autoencoder = create_deep_dense_ae()


def create_denoising_model(autoencoder):
    def add_noise(x):
        noise_factor = 0.2
        x = x + K.random_normal(x.get_shape(), 0.5, noise_factor)
        x = K.clip(x, 0., 1.)
        return x

    input_img = Input(batch_shape=(batch_size, 28, 28, 1))
    noised_img = Lambda(add_noise)(input_img)

    noiser = Model(input_img, noised_img, name="noiser")
    denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
    return noiser, denoiser_model


noiser, denoiser_model = create_denoising_model(autoencoder)
denoiser_model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])

history = denoiser_model.fit(x_train, x_train,
                   epochs=100,
                   batch_size=batch_size,
                   shuffle=True,
                   validation_data=(x_test, x_test))

n = 10

imgs = x_test[:batch_size]
noised_imgs = noiser.predict(imgs, batch_size=batch_size)
encoded_imgs = encoder.predict(noised_imgs[:n],  batch_size=n)
decoded_imgs = decoder.predict(encoded_imgs[:n], batch_size=n)

plot_digits(imgs[:n], noised_imgs, decoded_imgs)
plot_chart_loss(history)
plot_chart_accuracy(history)
