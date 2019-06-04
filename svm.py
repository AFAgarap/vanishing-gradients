import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 64
epochs = 200

class SVM:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.SGD(self.learning_rate, momentum=9e-1)
        self.weights = tf.Variable(tf.random.normal([4, 3]))
        self.biases = tf.Variable(tf.random.normal([3]))

    def predict(self, batch_features):
        logits = tf.add(tf.matmul(batch_features, self.weights), self.biases)
        return logits

    def loss_fn(self, logits, labels):
        regularizer_constant = 5e-1
        l2_norm = tf.reduce_mean(tf.square(self.weights))
        penalty_parameter = 5e-1
        squared_hinge_loss = tf.reduce_mean(tf.square(tf.maximum(0, 1 - logits * labels)))
        loss = (regularizer_constant * l2_norm) + (penalty_parameter * squared_hinge_loss)
        return loss

    def train_step(self, batch_features, batch_labels):
        with tf.GradientTape() as tape:
            logits = self.predict(batch_features)
            train_loss = self.loss_fn(logits=logits, labels=batch_labels)
        gradients = tape.gradient(train_loss, [self.weights, self.biases])
        self.optimizer.apply_gradients(zip(gradients, [self.weights, self.biases]))
        return train_loss

features, labels = load_iris().data, load_iris().target
features = features.astype(np.float32)
labels = tf.one_hot(labels, len(np.unique(labels)))
labels = labels.numpy()
labels[labels == 0] = -1

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, stratify=labels, test_size=0.30, shuffle=True)

dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
dataset = dataset.prefetch(train_features.shape[0] // batch_size)
dataset = dataset.shuffle(batch_size)
dataset = dataset.batch(batch_size=64, drop_remainder=True)

model = SVM(learning_rate=1e-3)

for epoch in range(epochs):
    epoch_loss = 0
    for step, (batch_features, batch_labels) in enumerate(dataset):
        train_loss = model.train_step(batch_features, batch_labels)
        epoch_loss += train_loss
    epoch_loss = np.mean(epoch_loss)
    
    if (epoch != 0) and (epoch + 1) % 10 == 0:
        print('epoch {}/{} : mean loss = {}'.format(epoch + 1, epochs, epoch_loss))

accuracy = tf.metrics.Accuracy()
predictions = tf.argmax(model.predict(test_features), 1)
accuracy(predictions, tf.argmax(test_labels, 1))
print('Test accuracy : {}'.format(accuracy.result()))
