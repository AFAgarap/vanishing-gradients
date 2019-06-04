import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(42)

def create_dataset(batch_size, data, onehot=False):
    if data == 'circles':
        features, labels = make_circles(n_samples=30000, noise=1e-1, random_state=42)
    elif data == 'moons':
        features, labels = make_moons(n_samples=30000, noise=1e-1, random_state=42)
    elif data == 'blobs':
        features, labels = make_blobs(n_samples=30000, random_state=42)

    features = features.astype(np.float32)
    labels = labels.astype(np.float32)

    train_features, test_features, train_labels, test_labels = train_test_split(features,
            labels,
            test_size=0.30,
            stratify=labels,
            random_state=42,
            shuffle=True)

    if onehot:
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

    dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    dataset = dataset.prefetch(train_features.shape[0] // batch_size)
    dataset = dataset.shuffle(batch_size * 2)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    return dataset, [test_features, test_labels]

def plot_gradients(model, splits=5):
    for split in range(splits):
        for index in range((len(model.hidden_layers) - 1) * 2):
            sns.kdeplot(np.array(model.gradient_means[1][1], dtype=np.float32), shade=True, color='r')