# Simple Feedforward Neural Network
This repository contains a simple feedforward neural network in written in Python using numpy. When the [script](https://github.com/thinks/feedforward-neural-network/blob/master/feedforward_neural_network.py) is run it trains a neural network and outputs the achieved error rate. The network is trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) data set (included in the [data](https://github.com/thinks/feedforward-neural-network/tree/master/data) folder). This data set consists of images of hand-written digits along with a label for each image giving the correct value, it is described in more detail [here](http://yann.lecun.com/exdb/mnist/). The task for the neural network is to learn how to recognize hand-written digits from a sub set of the images. Thereafter, the error rate is computed using previously unseen images.

To run the scripts simply clone this repository and run

```python
$ python feedforward_neural_network.py
```
