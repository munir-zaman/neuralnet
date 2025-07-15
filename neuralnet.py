import numpy as np
import random

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_deriv(z):
    sz = sigmoid(z)
    return sz * (1 - sz)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)/2

def mse_loss_deriv(y_pred, y_true):
    return (y_pred - y_true)


class NeuralNet():
    def __init__(self, layer_sizes, 
                 activation_func = sigmoid, 
                 activation_deriv = sigmoid_deriv, 
                 cost_func = mse_loss,
                 cost_deriv = mse_loss_deriv):
        
        self.total_layers = len(layer_sizes)

        # init weights and biases

        # weights[i] is the wieght from layer[i] to layer[i+1]
        # if w = weights[i], 
        # then w[j][k] is the weight between j-th neuron in layer[i] and k-th neuron in layer[i+1]
        # which means the weight matrix here is **transposed**
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) for i in range(self.total_layers-1)]

        # biases[i] is the bias for layer[i+1]
        self.biases = [np.random.rand(size) for size in layer_sizes[1:]] # ignore the first (input) layer

        # activation function
        # uses sigmoid by default
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        # uses mse by default
        self.cost_func = cost_func
        self.cost_deriv = cost_deriv

    def feedforward(self, input):
        activations = [input]
        z_values = []
        for i in range(self.total_layers-1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_values.append(z)
            activations.append(self.activation_func(z))
        return (activations, z_values)
    
    def SGD(self, training_data, epochs, learning_rate, batch_size):
        # training_data should be an array of (x, y) tuples
        # where x is the input vector and y is desired output vector
        # epochs is the number of times to train
        n = len(training_data)
        for i in range(epochs):
            # shuffle the data for each iteration
            random.shuffle(training_data)
            # create batches
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.learn(batch, learning_rate)

    def learn(self, batch, learning_rate = 0.1):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_w, delta_b = self.backprop(x, y)
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]

        self.weights = [w - learning_rate * (dw / len(batch)) for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - learning_rate * (db / len(batch)) for b, db in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # x is the input
        # y is the desired output
        
        # corrections
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activations, zs = self.feedforward(x)
        # delta of the last layer
        delta = self.cost_deriv(activations[-1], y) * self.activation_deriv(zs[-1])
        nabla_b[-1] = delta
        print(delta.shape)
        print(activations[-2].shape)
        nabla_w[-1] = activations[-2].reshape(activations[-2].shape[0], 1) @ delta.reshape(1, delta.shape[0])

        for i in range(2, self.total_layers):
            z = zs[-i]
            sp = self.activation_deriv(z)
            delta = (delta @ self.weights[-i+1].T) * sp
            nabla_b[-i] = delta
            print(activations[-i-1])
            nabla_w[-i] = activations[-i-1].reshape(activations[-i-1].shape[0], 1) @ delta.reshape(1, delta.shape[0])
            # nabla_w[-i] = activations[-i-1] @ delta
        
        return (nabla_w, nabla_b)
    
    def predict(self, x):
        activations, _ = self.feedforward(x)
        return activations[-1]

    def evaluate(self, test_data):
        correct = 0
        for x, y in test_data:
            pred = np.argmax(self.predict(x))
            target = np.argmax(y)
            if pred == target:
                correct += 1
        return correct / len(test_data)

