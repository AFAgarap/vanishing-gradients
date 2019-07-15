Avoiding the vanishing gradients problem by adding random noise and batch normalization 
===

_by Abien Fred Agarap, Joshua Raphaelle Cruzada, Gabrielle Marie Torres, Ralph Vincent Regalado, Charibeth Cheng, and Arnulfo Azcarraga, PhD_

## Abstract

The vanishing gradients problem is a problem that occurs in training neural networks with gradient-based learning methods and backpropagation -- the gradients will decrease to infinitesimally small values, thus preventing any update on the weights of a model. Since its discovery, several methods have been proposed to solve it. However, there have only been few attempts to compare them from both mathematical and empirical perspectives, thus the purpose of this work. We provide analyses through inspection of analytical gradients and their distribution, and classification performance of the neural networks. We also propose a novel method of adding Gaussian noise to gradients during training, coupled with batch normalization -- aimed to avoid the vanishing gradients problem. Our results show that using this approach, a neural net enjoys faster and better convergence -- having 11.25% higher test accuracy when compared to a baseline model.

## Results

In our experiments, we used the MNIST handwritten digits classification dataset for training and evaluating our neural networks. It consists of 60,000 training examples, and 10,000 test examples -- having 28x28 pixels per image in grayscale. We reshaped each image to 784-dimensional vector, scaled them by dividing each pixel with the maximum pixel value, and added random noise from Gaussian distribution having 0 mean and 0.05 standard deviation to prevent models from overfitting and
to elevate the difficulty to converge on the dataset.

### Experiment Setup

Experiments were done in a computer with Intel Core i5-6300HQ processor, 16GB RAM, and Nvidia GeForce 960M GPU with 4GB RAM.

### Improving Gradient Values 

We observed the distribution gradients of both baseline and experimental models during training, and the distributions for a neural network with logistic activation function are depicted in Figure 1. Since this _legacy_ activation function has the least maximum gradient value of 0.25, we considered observing the changes in its distribution to be noteworthy.

![](assets/mnist-logistic-dist.png)

**Figure 1. Gradient distribution over time of neural network with logistic activation function on MNIST dataset. _Top to bottom_: baseline model, model with gradient noise addition, and model with gradient noise addition + batch normalization.**

As we can see from the figure above, the gradient distribution of the model at hand drastically changes from the baseline configuration to the experimental configurations, i.e. from small value of -0.004 to 4. While this does not guarantee superior model performance, it does guarantee that there would be sufficient gradients to propagate through the neural network, thus avoiding the vanishing gradients problem.

### Classification Performance

![](assets/training-loss.png)

![](assets/training-accuracy.png)

## License
