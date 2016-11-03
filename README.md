# Simple Feedforward Neural Network
This repository contains a simple feedforward neural network in written in Python using numpy. When the [script](https://github.com/thinks/feedforward-neural-network/blob/master/feedforward_neural_network.py) is run it trains a neural network and outputs the achieved error rate. The network is trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset (included in the [data](https://github.com/thinks/feedforward-neural-network/tree/master/data) folder). This dataset consists of images of hand-written digits along with a label for each image giving the correct value; it is described in more detail [here](http://yann.lecun.com/exdb/mnist/). The task for the neural network is to learn how to recognize hand-written digits from these images. First the neural network is trained on a subset of the images. Thereafter, the error rate is computed using previously unseen images.

The [script](https://github.com/thinks/feedforward-neural-network/blob/master/feedforward_neural_network.py) is based on skeleton code provided by [Marco Kuhlmann](http://www.ida.liu.se/~marku61/).

To run the scripts simply clone this repository and run

```python
> python feedforward_neural_network.py
```

The output should look something like this (for 5 epochs)

```
Number of instances from training data: 60000
Number of instances from test data: 10000
Epoch 01, 60000 instances
Epoch 01, error rate = 5.62
Epoch 02, 60000 instances
Epoch 02, error rate = 3.69
Epoch 03, 60000 instances
Epoch 03, error rate = 3.59
Epoch 04, 60000 instances
Epoch 04, error rate = 3.48
Epoch 05, 60000 instances
Epoch 05, error rate = 2.81
```

The error rate is the generalization error, corresponding to the percentage of misclassified images in the training data set. In the output above, the network successfully classifies `100 - 2.81 = 97.19%` of the images.

A number of parameters can be changed, they are found in the settings section in the main function. Note that only sigmoid (logistic) activation functions are supported at the moment. Other parameters such as number of epochs and batch size may be changed freely.

## Future Work
* Currently only the simple logistic function can be used as activation in the hidden layers. It would be nice if [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), and possibly the hyperbolic tangent function, could be tested.
* Implement a similar structure using the [Theano](https://github.com/Theano/Theano) library.
