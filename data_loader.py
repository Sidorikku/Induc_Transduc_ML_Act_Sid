import tensorflow as tf
import os

# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    train_images_flat = train_images.reshape(-1, 28*28)
    test_images_flat = test_images.reshape(-1, 28*28)
    return train_images, train_labels, test_images, test_labels, train_images_flat, test_images_flat
