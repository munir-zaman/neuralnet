import numpy as np

INPUT_SIZE = 16
OUTPUT_SIZE = 16

# generate layers with random values
layer_sizes = [INPUT_SIZE, 5, 5, OUTPUT_SIZE]
layers = []
for i in range(0, len(layer_sizes)):
    if (i != 0) and (i != len(layer_sizes)-1):
        # random layers
        layers.append(np.random.rand(layer_sizes[i]))
    else:
        # set input and output layers to zeros
        layers.append(np.zeros(layer_sizes[i]))

# generate random wieghts
weights = []
for i in range(0, len(layer_sizes)-1):
    weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]))

# generate biases
biases = [np.random.rand(size) for size in layer_sizes[1:]]


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

activation_func = sigmoid
output_activation_func = softmax

def feed_forward():
    for i in range(len(weights)):
        z = layers[i] @ weights[i] + biases[i]
        if (i < len(weights) - 1):
            layers[i+1] = activation_func(z)
        else:
            layers[i+1] = output_activation_func(z)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

loss_func = mse_loss

input = np.array([
    [.5, .6, .1, .7],
    [.1, .2, .4, .3],
    [.05, .91, .8, .7],
    [.5, .5, .1, .2]
])
layers[0] = input.flatten()
feed_forward()
print(layers[-1])
