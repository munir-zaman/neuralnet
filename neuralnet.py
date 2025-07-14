import numpy as np

class NeuralNet():
    def __init__(self, layer_sizes):
        self.total_layers = len(layer_sizes)
        # init layers with random values
        self.layers = [np.random.rand(size) for size in layer_sizes]
        self.layers[0] = np.zeros(layer_sizes[0])
        self.layers[0] = np.zeros(layer_sizes[-1])

        # init weights and biases
        # weights[i] is the wieght from layer[i] to layer[i+1]
        # biases[i] is the bias for layer[i+1]
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(self.total_layers-1)]
        self.biases = [np.random.rand(size) for size in layer_sizes[1:]] # ignore the first (input) layer

    