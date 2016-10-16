# -*- coding: utf-8 -*-
"""
Feedforward neural network.

A simple feedforward network implemented using numpy. Trained and tested on the MNIST
data set consisting of images of hand-written digits.

Based on skeleton code provided by Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
"""

import gzip
import numpy as np
import struct

    
def read_image_and_label_data(image_archive_filename, 
                              label_archive_filename,
                              max_pairs=None):
    """
    Reads image and label data. This returns a list of pairs (x, y) where
    x is a 785-dimensional vector (a numpy.ndarray) representing
    the image and y is a 10-dimensional one-hot vector (a numpy.ndarray)
    representing the label.
    
    If max_pairs is None or less than one, reads all the images and labels in 
    the archives. Otherwise reads max_pairs number of images and labels 
    (or all available pairs).

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    
    with gzip.open(image_archive_filename) as img_file, \
            gzip.open(label_archive_filename) as lab_file:
        yield from ((vectorize_image(image), vectorize_label(label)) 
                    for image, label in zip(read_images(img_file, max_pairs),
                                            read_labels(lab_file, max_pairs)))

        
def read_images(input_stream, max_images=None):
    """
    Reads images from the input source. Returns a generator for those images.

    If max_images is None or less than one, reads all images from the input 
    source. Otherwise reads max_images number of images (or all available images).

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    
    magic = struct.unpack('>BBBB', input_stream.read(4))
    assert magic[0] == 0 and magic[1] == 0 and magic[2] == 8 and magic[3] == 3

    img_count = struct.unpack('>i', input_stream.read(4))[0]
    if max_images is not None and max_images >= 1:
        img_count = np.minimum(img_count, max_images)

    row_count = struct.unpack('>i', input_stream.read(4))[0]
    col_count = struct.unpack('>i', input_stream.read(4))[0]
    pixel_count = row_count * col_count
    for i in range(img_count):
        yield struct.unpack('>%dB' % pixel_count, input_stream.read(pixel_count))

        
def read_labels(input_stream, max_labels=None):
    """
    Reads labels from the input source. Returns a generator for those labels.

    If max_labels is None or less than one, reads all labels from the input 
    source. Otherwise reads max_labels number of labels (or all available labels).

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    
    magic = struct.unpack('>BBBB', input_stream.read(4))
    assert magic[0] == 0 and magic[1] == 0 and magic[2] == 8 and magic[3] == 1

    label_count = struct.unpack('>i', input_stream.read(4))[0]
    if max_labels is not None and max_labels >= 1:
        label_count = np.minimum(label_count, max_labels)

    for i in range(label_count):
        yield struct.unpack('>B', input_stream.read(1))[0]
        

def vectorize_image(image):
    """
    Maps an MNIST image to a vector (a numpy.ndarray). An MNIST image
    is a 784-component tuple of integers between 0 and 255,
    representing greyscale values. The resulting vector is a
    785-dimensional column vector: The first component of the vector
    is 1; the remaining 784 components represent the greyscale
    values. Each greyscale value is normalized to the interval [0, 1].

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    
    image = (255,) + image  # First component will become 1.
    return np.reshape(np.array(image, dtype=float), (785, 1)) / 255


def vectorize_label(label):
    """
    Maps an MNIST label to a vector (a "numpy.ndarray"). An MNIST label
    is an integer "i" between 0 and 9. The resulting vector is a
    10-dimensional column vector with a 1.0 in the "i"th position and
    zeros elsewhere.

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    
    x = np.zeros((10, 1))
    x[label] = 1.0
    return x

    
class Network(object):
    """
    TODO!
    """
    def __init__(self, 
                 sizes, 
                 activation_func, 
                 activation_func_derivative, 
                 initial_weights_func, 
                 use_softmax=True):
        """
        Constructor.
        Create and initialize weights, setup structures for storing 
        intermediate values while doing feedforward. These values are re-used
        when backpropagating.
        """
        
        self.sizes = sizes
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.use_softmax = use_softmax

        # Must have at least one hidden layer.
        # Initialize weights.
        assert(len(self.sizes) >= 3) 
        self.w = [initial_weights_func(n_out, n_in) 
                  for n_out, n_in in zip(self.sizes[1:], self.sizes[:-1])]
                
        # Note that we have a dummy entry in the inputs (z) list here.
        # The input layer has no inputs, but we have an entry for
        # it so that we can use the same index for input/output for
        # a given layer. We set that entry to null to make sure we 
        # never use it.
        self.z = [np.zeros((s, 1)) if i > 0 else None 
                  for i, s in enumerate(self.sizes)]
        self.y = [np.zeros((s, 1)) for s in self.sizes]

    def feedforward(self, x):
        """
        TODO!
        """

        # Store the output of the input layer simply as the example data.
        self.y[0] = np.array(x)
        
        # Compute input/output for hidden layers.
        n_hidden = len(self.w) - 1
        for i in range(n_hidden):
            # First compute weighted sum of inputs (outputs from previous layer).
            # Then run the activation function on these values to produces the 
            # outputs for this hidden layer.
            # Store input/output values for back propagation purposes.
            self.z[i + 1] = np.dot(self.w[i], self.y[i])
            self.y[i + 1] = self.activation_func(self.z[i + 1])            

        # Output layer.    
        # Only softmax or sigmoid activation functions make sense 
        # for the output layer.
        self.z[-1] = np.dot(self.w[-1], self.y[-2])    
        if self.use_softmax:
            self.y[-1] = np.divide(np.exp(self.z[-1]), np.sum(np.exp(self.z[-1])))
        else:
            self.y[-1] = np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, self.z))))
        
        # Return the output layer values.
        return self.y[-1]

    def update(self, batch, eta):
        """
        TODO!
        """

        # Extract image data from batch.
        x = [batch[i][0] for i in range(len(batch))]
        y_target = [batch[i][1] for i in range(len(batch))]
        sum_grad_w = [np.zeros(w.shape) for w in self.w]
        for i in range(len(x)):
            # The output layer is already stored in self.y[-1] so we don't 
            # need to store the return value here.
            self.feedforward(x[i])
            
            # Compute and accumulate error gradients.
            grad_w = self.backpropagate(y_target[i])
            for j in range(len(grad_w)):
                sum_grad_w[j] += grad_w[j]
            
        # Update weights before experiencing next batch.
        for i in range(len(self.w)):
            self.w[i] = np.subtract(self.w[i], np.multiply(eta, sum_grad_w[i]))

    def backpropagate(self, t):
        """
        TODO!
        """
        
        # Compute deltas for the output layer. These deltas 
        # become input to the hidden layers.
        # Assumes activation function in the output layer is softmax or sigmoid.
        delta = np.subtract(self.y[-1], t)            

        # Compute deltas and gradients for hidden layers.
        grad = [np.zeros(w.shape) for w in self.w]
        for i in range(len(self.w) - 1, 0, -1):
            # Compute the gradients for this weight layer using 
            # deltas from previous layer.
            grad[i] = np.dot(delta, np.transpose(self.y[i]))
            
            # Compute deltas for this layer using deltas from 
            # the layer directly above.      
            # Compute the activation function and its derivate
            # for the inputs to this layer.
            dy_dz = self.activation_func_derivative(self.z[i])
            h_sums = np.dot(np.transpose(delta), self.w[i])
            np.resize(delta, (dy_dz.shape[0] * h_sums.shape[0], 1))
            delta = np.multiply(dy_dz, np.transpose(h_sums))
        
        # Compute gradients for input layer (using deltas from 
        # first hidden layer).
        grad[0] = np.dot(delta, np.transpose(self.y[0]))
        
        return grad

        
def sigmoid(z):
    """
    z should be a column or row vector.
    """

    return np.divide(1.0, np.add(1.0, np.exp(np.multiply(-1.0, z))))
    
    
def sigmoid_derivative(z):
    """
    z should be a column or row vector.
    """

    s = sigmoid(z)
    return np.multiply(s, np.subtract(1.0, s))
    
    
def sigmoid_initial_weights(n_out, n_in):
    """
    Initialize weights to be in the range suggested by Glorot and Bengio (2010) 
    for sigmoid activation functions. This ensures that in early training each 
    neuron operates on values that are in the non-flat areas of the sigmoid 
    function.    
    """
    
    return np.random.uniform(low=-4 * np.sqrt(6.0 / (n_in + n_out))[0],
                             high=4 * np.sqrt(6.0 / (n_in + n_out))[0],
                             size=(n_out, n_in))
  
    
def tanh(z):
    """ 
    z should be a column or row vector. 
    """
    e_pos = np.exp(z)
    e_neg = np.exp(np.multiply(-1.0, z))
    return np.divide(np.subtract(e_pos, e_neg), np.add(e_pos, e_neg))
    
    
def tanh_derivative(z):
    """ 
    z should be a column or row vector. 
    """
    f_z = tanh(z)
    return np.subtract(1.0, np.multiply(f_z, f_z))

    
def tanh_initial_weights(n_out, n_in):
    """
    Initial weights for tanh activation function.
    """
    return np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out))[0],
                             high=np.sqrt(6.0 / (n_in + n_out))[0],
                             size=(n_out, n_in))


def relu(z):
    """ 
    z should be a column or row vector. 
    """
    return np.maximum(0.0, z)


def relu_derivative(z):
    """ 
    z should be a column or row vector. 
    """
    return np.where(z <= 0.0, 0.0, 1.0)    


def relu_initial_weights(n_out, n_in):
    """
    TODO!
    """
    return np.random.uniform(low=-np.sqrt(6.0 / (n_in + n_out))[0],
                             high=np.sqrt(6.0 / (n_in + n_out))[0],
                             size=(n_out, n_in))
    

def train_network(net, data, epoch_count, batch_size, eta, error_rate_func=None):
    """
    TODO!

    Function courtesy of Marco Kuhlmann (http://www.ida.liu.se/~marku61/)
    """
    n = len(data)
    for e in range(epoch_count):
        np.random.shuffle(data)
        for k in range(0, n, batch_size):
            mini_batch = data[k:k+batch_size]
            net.update(mini_batch, eta)
            print("\rEpoch %02d, %05d instances" % (e, k + batch_size), end="")
        print()
        if error_rate_func:
            error_rate = error_rate_func(net)
            print("Epoch %02d, error rate = %.2f" % (e, error_rate * 100))
            
    
def get_error_rate(net, error_data):
    """
    Returns a value in the range [0, 1], where zero corresponds to the network
    predicting the correct value for every instance in test_data, and one
    corresponds to the network predicting the incorrect value for every 
    instance in test_data.
    """
    
    assert len(error_data) > 0
    error_count = 0.0
    for image, label in error_data:
        y_hat = net.feedforward(image)
        if np.argmax(y_hat, axis=0) != np.argmax(label, axis=0):
            error_count += 1.0
    
    return error_count / len(test_data)    
       
    
def network_sizes(input_layer_size, output_layer_size, hidden_layer_sizes):
    """
    Return a list in the format:
    [input_layer_size,
     hidden_layer_size[0], hidden_layer_size[1], ...,
     output_layer_size]
    """
    
    sizes = [input_layer_size]
    for hidden_layer_size in hidden_layer_sizes:
        sizes.append(hidden_layer_size)
    sizes.append(output_layer_size)  
    return sizes        
    
    
if __name__ == "__main__":
    """
    Entry point.
    
    Read data, train network, print error rate error.
    """
    
    TRAINING_COUNT = 30000
    training_images_archive_filename = "./data/train-images-idx3-ubyte.gz"
    training_labels_archive_filename = "./data/train-labels-idx1-ubyte.gz"
    training_data = list(read_image_and_label_data(training_images_archive_filename, 
                                                   training_labels_archive_filename,
                                                   max_pairs=TRAINING_COUNT))
    print("Number of instances from training data: {0}".format(len(training_data)))
    
    TEST_COUNT = 5000
    test_images_archive_filename = "./data/t10k-images-idx3-ubyte.gz"
    test_labels_archive_filename = "./data/t10k-labels-idx1-ubyte.gz"
    test_data = list(read_image_and_label_data(test_images_archive_filename, 
                                               test_labels_archive_filename,
                                               max_pairs=TEST_COUNT))
    print("Number of instances from test data: {0}".format(len(test_data)))

    # Make sure we always use the same seed so that we can compare results 
    # between runs.
    np.random.seed(1234)
    
    # Settings.
    EPOCH_COUNT = 2
    BATCH_SIZE = 10
    LEARNING_RATE = 0.1
    INPUT_LAYER_SIZE = 785
    OUTPUT_LAYER_SIZE = 10
    HIDDEN_LAYER_SIZES = [100, 100]
    USE_SOFTMAX = True
    ACTIVATION_FUNC = sigmoid
    ACTIVATION_FUNC_DERIVATIVE = sigmoid_derivative
    INITIAL_WEIGHTS_FUNC = sigmoid_initial_weights

    network = Network(sizes=network_sizes(INPUT_LAYER_SIZE,
                                          OUTPUT_LAYER_SIZE,
                                          HIDDEN_LAYER_SIZES),
                      activation_func=ACTIVATION_FUNC,
                      activation_func_derivative=ACTIVATION_FUNC_DERIVATIVE,
                      initial_weights_func=INITIAL_WEIGHTS_FUNC,
                      use_softmax=USE_SOFTMAX)
    train_network(network,
                  training_data, 
                  n_epochs=EPOCH_COUNT,
                  batch_size=BATCH_SIZE,
                  eta=LEARNING_RATE,
                  error_rate_func=lambda n: get_error_rate(n, test_data))
