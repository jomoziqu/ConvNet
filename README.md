# ConvNet
This project aims to implement a simple two-layer convolutional neural network (CNN) in Python, with a specific architecture and set of parameters. The goal is to compute the output of the network, along with the gradients of the hidden layer and output with respect to the weight parameters.

## Architecture

The CNN has five inputs (x1, ..., x5), four hidden nodes (z1, ..., z4), and one output (y). The hidden layer and output are computed using ReLU activations. The network is defined as follows:

w1 = 1.2,
w2 = -0.2,
v1 = -0.3,
v2 = 0.6,
v3 = 1.3,
v4 = -1.5

## Function Interface

The CNN implementation is a single Python function named convnet that takes a list of five numerical inputs x and returns the output y as a number of type dual and the hidden layer outputs z as a list of four numbers of type dual.

The function interface is as follows:

y, z = convnet(x),
where x is a list of five numerical inputs.
