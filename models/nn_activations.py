from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import tensorflow as tf


class NeuralNet:
    def __init__(self, layers, initialization):
        self.loss_values = []
        self.weights = []
        self.layers = layers
        self.num_layers = len(layers)
        self.initialization = initialization

    def initialize_params(self):
        if self.initialization == 'zeros':
            for layer in range(1, self.num_layers):
                self.weights.append(tf.Variable(tf.zeros([self.layers[layer], self.layers[layer - 1]])))
        elif self.initialization == 'ones':
            for layer in range(1, self.num_layers):
                self.weights.append(tf.Variable(tf.ones([self.layers[layer], self.layers[layer - 1]])))
        elif self.initialization == 'normal':
            for layer in range(1, self.num_layers):
                self.weights.append(tf.Variable(tf.random.normal([self.layers[layer], self.layers[layer - 1]])))
        elif self.initialization == 'xavier':
            initializer = tf.keras.initializers.glorot_normal()
            for layer in range(1, self.num_layers):
                self.weights.append(tf.Variable(initializer([self.layers[layer], self.layers[layer - 1]])))

    def forward_prop(self, batch_features):
        activations = []
        linear_activations = []
        activations.append(tf.transpose(batch_features))
        for layer in range(1, self.num_layers):
            linear_activations.append(tf.matmul(self.weights[layer - 1], activation[layer - 1]))
            if layer != self.num_layers - 1:
                activations.append(tf.nn.relu(linear_activation[layer - 1]))
        return tf.transpose(linear_activations[self.num_layers - 2])

    @tf.function
    def predict(self, batch_features):
        logits = self.forward_prop(batch_features)
        return logits

    def train(self, train_features, train_labels, test_features, test_labels, epochs=10, batch_size=128):
        self.initialize_params()
        num_examples = train_features.shape[0]

        optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in range(int(num_examples / batch_size)):
                batch_features = train_features[(batch * batch_size) : (batch * batch_size + batch_size)]
                batch_labels = train_labels[(batch * batch_size) : (batch * batch_size + batch_size)]

                with tf.GradientTape() as tape:
                    logits = self.forward_prop(batch_features)
                    batch_loss = tf.losses.categorical_crossentropy(batch_labels, logits, from_logits=True)
                    batch_loss = tf.reduce_mean(batch_loss)
                gradients = tape.gradient(batch_loss, self.weights)
                optimizer.apply_gradients(zip(gradients, self.weights))

                epoch_loss += batch_loss
            self.loss_values.append(epoch_loss)
            if epoch != 0 and (epoch + 1) % 10 == 0:
                print('Epoch {}/{}. Loss : {}'.format(epoch + 1, epochs, epoch_loss))
        predictions = self.predict(test_features)
        accuracy = tf.metrics.Accuracy()
        accuracy(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
        print('Test Accuracy : {}'.format(accuracy.numpy()))
