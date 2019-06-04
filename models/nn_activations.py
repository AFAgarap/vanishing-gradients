# Exploration of Vanishing Gradients and its solutions
# Copyright (C) 2019  Abien Fred Agarap, Joshua Cruzada
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Implementation of Neural Network with different activations"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')

tf.config.gpu.set_per_process_memory_growth(True)

class NeuralNet:
    def __init__(self, layers, activation):
        self.weights = []
        self.layers = layers
        self.num_layers = len(layers)
        self.activation = activation

    def initialize_params(self):
        for layer in range(1, self.num_layers):
            self.weights.append(tf.Variable(tf.random.normal([self.layers[layer], self.layers[layer - 1]])))

    def forward_prop(self, batch_features):
        activations = []
        linear_activations = []
        activations.append(tf.transpose(batch_features))
        for layer in range(1, self.num_layers):
            linear_activations.append(tf.matmul(self.weights[layer - 1], activations[layer - 1]))
            if layer != self.num_layers - 1:
                if self.activation == 'relu':
                    activations.append(tf.nn.relu(linear_activations[layer - 1]))
                elif self.activation == 'sigmoid':
                    activations.append(tf.nn.sigmoid(linear_activations[layer - 1]))
                elif self.activation == 'tanh':
                    activations.append(tf.nn.tanh(linear_activations[layer - 1]))
                elif self.activation == 'leaky_relu':
                    activations.append(tf.nn.leaky_relu(linear_activations[layer - 1]))
                elif self.activation == 'swish':
                    activations.append(tf.nn.sigmoid(linear_activations[layer - 1]) * linear_activations[layer - 1])
        return tf.transpose(linear_activations[self.num_layers - 2])

    @tf.function
    def predict(self, batch_features):
        logits = self.forward_prop(batch_features)
        return logits

    def train(self, dataset, epochs=10):
        self.initialize_params()

        optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        writer = tf.summary.create_file_writer('tmp')

        with writer.as_default():
            with tf.summary.record_if(True):
                for epoch in range(epochs):
                    epoch_loss = 0
                    epoch_accuracy = []
                    for step, (batch_features, batch_labels) in enumerate(dataset):
                        with tf.GradientTape() as tape:
                            logits = self.forward_prop(batch_features)
                            batch_loss = tf.losses.categorical_crossentropy(batch_labels, logits, from_logits=True)
                            batch_loss = tf.reduce_mean(batch_loss)
                        gradients = tape.gradient(batch_loss, self.weights)
                        optimizer.apply_gradients(zip(gradients, self.weights))

                        accuracy = tf.metrics.Accuracy()
                        accuracy(tf.argmax(self.predict(batch_features), 1), tf.argmax(batch_labels, 1))

                        epoch_loss += batch_loss
                        epoch_accuracy.append(accuracy.result())
                    epoch_loss = np.mean(epoch_loss)
                    epoch_accuracy = np.mean(epoch_accuracy)

                    tf.summary.scalar('loss', epoch_loss, step=step)
                    tf.summary.scalar('accuracy', epoch_accuracy, step=step)

                    if epoch != 0 and (epoch + 1) % 10 == 0:
                        print('Epoch {}/{}. Loss : {}, Accuracy : {}'.format(epoch + 1, epochs, epoch_loss, epoch_accuracy))

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()
train_features = train_features.reshape(-1, 784).astype(np.float32) / 255.
test_features = test_features.reshape(-1, 784).astype(np.float32) / 255.
train_labels = tf.one_hot(train_labels, len(np.unique(train_labels)))
test_labels = tf.one_hot(test_labels, len(np.unique(test_labels)))

dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
dataset = dataset.prefetch(train_features.shape[0] // 128)
dataset = dataset.shuffle(256)
dataset = dataset.batch(batch_size=256, drop_remainder=True)

model = NeuralNet([784, 128, 128, 10], activation='tanh')
model.train(dataset, epochs=100)

predictions = model.predict(test_features)
accuracy = tf.metrics.Accuracy()
accuracy(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
print('Test Accuracy : {}'.format(accuracy.result()))

