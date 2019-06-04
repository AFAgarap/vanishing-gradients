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
__author__ = 'Joshua Raphaelle R. Cruzada'

import tensorflow as tf
import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layers, activation='relu', final_activation='softmax'):
        self.hidden_layers = hidden_layers
        self.weights = []
        self.biases = []
        self.optimizer = tf.optimizers.Adam(3e-4)
        self.activation = activation
        self.final_activation = final_activation
        self.gradient_means = [[] for layer in range(0, (len(hidden_layers) - 1) * 2)]
        
    def initialize_params(self):
        for layer in range(1, len(self.hidden_layers)):
            self.weights.append(tf.Variable(tf.random.normal([self.hidden_layers[layer - 1], self.hidden_layers[layer]])))
            self.biases.append(tf.Variable(tf.random.normal([self.hidden_layers[layer]])))
    
    @tf.function
    def predict(self, batch_features):
        logits = batch_features
        for index in range(0, len(self.weights)):
            logits = tf.matmul(logits, self.weights[index]) + self.biases[index]
            if((index != len(self.weights) - 1)):
                if(self.activation == 'relu'):
                    logits = tf.nn.relu(logits)
                elif(self.activation == 'sigmoid'):
                    logits = tf.nn.sigmoid(logits)
                elif(self.activation == 'tanh'):
                    logits = tf.nn.tanh(logits)
                elif(self.activation == 'leaky_relu'):
                    logits = tf.nn.leaky_relu(logits)
                elif(self.activation == 'swish'):
                    logits = tf.nn.swish(logits)
        return logits
    
    def loss_fn(self, logits, labels):
        if(self.final_activation == 'softmax'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            l2_norm = tf.reduce_mean(tf.square(self.weights[len(self.weights) - 1]))
            penalty_parameter = 5e-1
            regularizer_constant = 5e-1
            squared_hinge_loss = tf.reduce_mean(tf.square(tf.maximum(0, 1 - logits * labels)))
            loss = (regularizer_constant * l2_norm) + (penalty_parameter * squared_hinge_loss)
        return loss
        
    
    def train_step(self, batch_features, batch_labels):
        with tf.GradientTape() as tape:
            logits = self.predict(batch_features)
            loss = self.loss_fn(logits, batch_labels)
            gradients = tape.gradient(loss, [*self.weights, *self.biases])
            self.optimizer.apply_gradients(zip(gradients, [*self.weights, *self.biases]))
            return loss, gradients

    def train(self, dataset, epochs=1, batch_size=128):
        self.initialize_params()
        for epoch in range(epochs):
            epoch_loss = 0
            temp_gradients = [[] for layer in range(0, (len(self.hidden_layers) - 1) * 2)]
            for step, (batch_features, batch_labels) in enumerate(dataset):
                train_loss, gradients = self.train_step(batch_features, batch_labels)
                epoch_loss += train_loss
                for gradient in range(len(gradients)):
                    temp_gradients[gradient].append(np.mean(gradients[gradient].numpy()))
            for gradient in range(len(temp_gradients)):
                self.gradient_means[gradient].append(temp_gradients[gradient])
            if epoch != 0 and (epoch + 1) % 10 == 0:
                print('Epoch {}/{}. Loss : {}'.format(epoch + 1, epochs, np.mean(epoch_loss)))


(train_features, train_labels), (test_features, test_labels)=tf.keras.datasets.mnist.load_data()
train_features = train_features.reshape(-1, 784).astype(np.float32) / 255.
test_features = test_features.reshape(-1, 784).astype(np.float32) / 255.
train_labels = tf.one_hot(train_labels, len(np.unique(train_labels)))
test_labels = tf.one_hot(test_labels, len(np.unique(test_labels)))
train_labels = train_labels.numpy()
test_labels = test_labels.numpy()
train_labels[train_labels == 0] = -1
test_labels[test_labels == 0] = -1

dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
dataset = dataset.prefetch(train_features.shape[0] // 64)
dataset = dataset.shuffle(buffer_size=64)
dataset = dataset.batch(batch_size=64, drop_remainder=True)

model = NeuralNetwork([784, 128, 128, 10], 'relu', 'svm')
gradient_means = model.train(dataset, epochs=50)

predictions = model.predict(test_features)
accuracy = tf.metrics.Accuracy()
accuracy(tf.argmax(predictions, 1), tf.argmax(test_labels, 1))
print('Test Accuracy : {}'.format(accuracy.result()))