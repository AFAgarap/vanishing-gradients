# Gradient noise addition with batch norm
# Copyright (C) 2019  Abien Fred Agarap
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
"""Implementation of feed-forward neural net"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.num_layers = kwargs['num_layers']
        self.neurons = kwargs['neurons']
        self.hidden_layers = []
        self.activation = kwargs['activation']
        for index in range(self.num_layers):
            self.hidden_layers.append(
                    tf.keras.layers.Dense(
                        units=self.neurons[index], activation=self.activation
                        )
                    )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'], activation=tf.nn.softmax
                )
        self.optimizer = tf.optimizers.SGD(
                learning_rate=1e-1,
                momentum=9e-1,
                decay=1e-6,
                nesterov=True
                )

    @tf.function
    def call(self, features):
        activations = []
        for index in range(self.num_layers):
            if index == 0:
                activations.append(self.hidden_layers[index](features))
            else:
                activations.append(
                        self.hidden_layers[index](
                            activations[index - 1]
                            )
                        )
        output = self.output_layer(activations[-1])
        return output


def loss_fn(logits, labels):
    return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
                )
            )


def train_step(model, loss, features, labels, epoch):
    with tf.GradientTape() as tape:
        logits = model(features)
        train_loss = loss(logits, labels)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss, gradients
