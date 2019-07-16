"""Implementation of gradient noise addition"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import argparse
import tensorflow as tf
import time

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.random.set_seed(42)


class NeuralNet(tf.keras.Model):
    def __init__(self, units):
        super(NeuralNet, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=units, activation=self.swish)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=units, activation=self.swish)
        self.output_layer = tf.keras.layers.Dense(units=10)
        self.optimizer = tf.optimizers.SGD(learning_rate=3e-4, momentum=9e-1)

    @tf.function
    def call(self, batch_features):
        activations = self.hidden_layer_1(batch_features)
        activations = self.hidden_layer_2(activations)
        return self.output_layer(activations)

    def swish(self, z):
        return z * tf.nn.sigmoid(z)


def loss_fn(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def train_step(model, loss, features, labels, epoch):
    with tf.GradientTape() as tape:
        logits = model(features)
        train_loss = loss(logits, labels)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    stddev = 1 / ((1 + epoch)**0.55)
    gradients = [tf.add(gradient, tf.random.normal(stddev=stddev, mean=0., shape=gradient.shape)) for gradient in gradients]
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss, gradients


def plot_gradients(gradients, step):
    for index, gradient in enumerate(gradients):
        if len(gradient.shape) == 1:
            tf.summary.histogram('histogram/{}-bias-grad'.format(index), gradient, step)
        elif len(gradient.shape) != 1:
            tf.summary.histogram('histogram/{}-weights-grad'.format(index), gradient, step)


def train(model, loss_fn, dataset, epochs=10):

    writer = tf.summary.create_file_writer('tmp/{}-swish-injection-fashion_mnist'.format(time.asctime()))

    with writer.as_default():
        with tf.summary.record_if(True):
            step = 0
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_accuracy = []
                for batch_features, batch_labels in dataset:

                    batch_loss, train_gradients = train_step(model, loss_fn, batch_features, batch_labels, epoch)

                    accuracy = tf.metrics.Accuracy()
                    accuracy(tf.argmax(model(batch_features), 1), tf.argmax(batch_labels, 1))

                    epoch_loss += batch_loss
                    epoch_accuracy.append(accuracy.result())
                    plot_gradients(train_gradients, step)

                    step += 1

                epoch_loss = tf.reduce_mean(epoch_loss)
                epoch_accuracy = tf.reduce_mean(epoch_accuracy)

                tf.summary.scalar('loss', epoch_loss, step=step)
                tf.summary.scalar('accuracy', epoch_accuracy, step=step)

                if epoch != 0 and (epoch + 1) % 10 == 0:
                    print('Epoch {}/{}. Loss : {}, Accuracy : {}'.format(epoch + 1, epochs, epoch_loss, epoch_accuracy))


def main(arguments):
    batch_size = 1024
    epochs = 100

    (train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    train_features = train_features.reshape(-1, 784) / 255.
    train_features += tf.random.normal(stddev=5e-2, mean=0., shape=train_features.shape)
    test_features = test_features.reshape(-1, 784) / 255.

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_dataset = train_dataset.prefetch(batch_size * 2)
    train_dataset = train_dataset.shuffle(batch_size * 2)
    train_dataset = train_dataset.batch(batch_size, True)

    model = NeuralNet(units=512)
    start_time = time.time()
    train(model, loss_fn, train_dataset, epochs=epochs)
    print('training time : {}'.format(time.time() - start_time))

    accuracy = tf.metrics.Accuracy()
    accuracy(tf.argmax(model(test_features), 1), tf.argmax(test_labels, 1))
    print('test accuracy : {}'.format(accuracy.result()))


if __name___ == '__main__':
    arguments = parse_args()
    main(arguments)
