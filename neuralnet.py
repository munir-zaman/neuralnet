import numpy as np
import random
import pickle

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_deriv(z):
    sz = sigmoid(z)
    return sz * (1.0 - sz)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def mseloss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)/2

def mseloss_deriv(y_pred, y_true):
    return (y_pred - y_true)


def list2rowvector(l):
    arr = np.array(l)
    arr = arr.reshape(1, arr.size)
    return arr

def list2colvector(l):
    arr = np.array(l)
    arr = arr.reshape(arr.size, 1)
    return arr

def vector2list(arr):
    l = arr.reshape(arr.size)
    return l

class NeuralNet():
    def __init__(self,
                 layer_sizes,
                 activation_func = sigmoid, 
                 activation_deriv = sigmoid_deriv, 
                 cost_func = mseloss,
                 cost_deriv = mseloss_deriv):
        
        self.layer_size = layer_sizes
        self.total_layers = len(layer_sizes)
        self.activation_func = activation_func
        self.activation_deriv = activation_deriv
        self.cost_func = cost_func
        self.cost_deriv = cost_deriv

        # randomly initilize weights and biases

        # weights are transposed 
        # shape of w[i] = (layer_sizes[i], layer_size[i+1])
        # w_l is w[l-1]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.total_layers-1)]
        # biases are also transposed
        # shape of b[i] = (1, layer_szie[i+1])
        # b_l is b[l-1]
        self.biases = [np.random.randn(1, n) for n in layer_sizes[1:]]

        # NOTE: since everything is transposed, remember to change the order of operation
        # for matrix multiplications e.g. do a @ W instead of W @ a
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.weights, self.biases = pickle.load(f)

    def forwardpass(self, input):
        # a_l is a[l]
        activations = [np.zeros((1, n)) for n in self.layer_size]
        activations[0] = list2rowvector(input)

        # z_l is z[l-1]
        z_values = [np.zeros((1, n)) for n in self.layer_size[1:]]

        for l in range(1, self.total_layers):
            # z_l = a_(l-1) @ w_l + b_l
            z_values[l-1] = activations[l-1] @ self.weights[l-1] + self.biases[l-1]
            activations[l] = self.activation_func(z_values[l-1])

        return activations, z_values
    
    def backprop(self, x, y):
        # x is the input
        # y is the desired output

        # corrections
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forwardpass
        activations, zs = self.forwardpass(x)

        # NOTE: consider the shape of x and y
        # we dont need to turn y into a row vector
        
        # the delta for the last layer
        delta = self.cost_deriv(activations[-1], y) * self.activation_deriv(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = activations[-2].T @ delta

        # move backward
        for i in range(2, self.total_layers):
            z = zs[-i]
            deriv_z = self.activation_deriv(z)
            delta = (delta @ self.weights[-i+1].T) * deriv_z
            nabla_b[-i] = delta
            nabla_w[-i] = activations[-i-1].T @ delta

        return (nabla_w, nabla_b)
    
    def learn(self, batch, learning_rate = 3):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            delta_w, delta_b = self.backprop(x, y)
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
        
        self.weights = [w-(learning_rate/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(learning_rate/len(batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]


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

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forwardpass(x)[0][-1]), np.argmax(y)) for x, y in test_data]
        correct = sum(int(pred == label) for pred, label in test_results)
        print(f"Accuracy: {correct}/{len(test_data)} ({100 * correct / len(test_data):.2f}%)")
        return correct
