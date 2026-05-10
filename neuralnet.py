import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_deriv(z):
    sz = sigmoid(z)
    return sz * (1.0 - sz)

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

# loss functions 
# and their derivatives with respect to activation (or y_pred)

# mse loss

def mseloss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)/2

def mseloss_deriv(y_pred, y_true):
    """
        returns an array containing the derivatives of the loss function 
        with respect to each activation
    """
    return (y_pred - y_true)

# cross entropy

# adding this to y_pred 
# so we dont get dividsion by zero T.T
y_pred_epsilon = 0.00001

def cross_entropy(y_pred, y_true):
    return - np.sum ( ( y_true * np.log(y_pred + y_pred_epsilon) 
                        + (1 - y_true) * np.log(1 - y_pred + y_pred_epsilon) ) )

def cross_entropy_deriv(y_pred, y_true):
    """
        returns an array containing the derivatives of the loss function 
        with respect to each activation
    """
    return - ( y_true / (y_pred + y_pred_epsilon) 
                - (1 - y_true) / (1 - y_pred + y_pred_epsilon) )

# some helper functions

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
                 cost_func = cross_entropy,
                 cost_deriv = cross_entropy_deriv):
        
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
        """
            uses pickle to save the weights and biases to a file
        """
        with open(filename, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load_model(self, filename):
        """
            uses pickle to load saved weights and biases from a file
        """
        with open(filename, 'rb') as f:
            self.weights, self.biases = pickle.load(f)

    def forwardpass(self, input):
        """
            `input` should be a flattened array of shape (1, 28*28) (assuming we are working with MNIST) 
            with values between 0 and 1 (normalized). 
            this method outputs a tuple containing activation and z_values respectively

            Example usage:
            ==============
                img = Image.open(f"../my-test/two.bmp").convert('L')  # force 8-bit grayscale
                arr = np.array(img).astype(np.float64) / 255.0 # normalize rgb values
                np.resize(arr, (1, 28*28)) # resize
                print(np.argmax(mnist_net.forwardpass(arr)[0][-1])) # print the output
        """
        # a_l is a[l]
        activations = [np.zeros((1, n)) for n in self.layer_size]
        activations[0] = list2rowvector(input)

        # z_l is z[l-1]
        z_values = [np.zeros((1, n)) for n in self.layer_size[1:]]

        for l in range(1, self.total_layers):
            # z_l = a_(l-1) @ w_l + b_l
            z_values[l-1] = activations[l-1] @ self.weights[l-1] + self.biases[l-1]
            activations[l] = self.activation_func(z_values[l-1])

        # apply softmax on the last layer
        # to turn this into a probability distribution
        softmax(activations[-1])

        return activations, z_values
    
    def backprop_delta(self, activations, zs, y):
        # the delta for the last layer
        delta = [np.array([])] * (self.total_layers - 1)
        delta[-1] = self.cost_deriv(activations[-1], y) * self.activation_deriv(zs[-1])
        # move backward
        for i in range(2, self.total_layers):
            z = zs[-i]
            deriv_z = self.activation_deriv(z)
            delta[-i] = (delta[-i + 1] @ self.weights[-i + 1].T) * deriv_z

        return delta

    def backprop(self, x, y):
        """
            `x` is passed to `self.forwardpass` to get the activations and z_values for x. 
            see `self.forwardpass` for what the input `x` should be.

            this method outputs a tuple containing nabla_w and nabla_b respectively.
        """
        # x is the input
        # y is the desired output

        # corrections
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forwardpass
        activations, zs = self.forwardpass(x)
        # get the deltas
        delta = self.backprop_delta(activations, zs, y)

        # NOTE: consider the shape of x and y
        # we dont need to turn y into a row vector
        
        nabla_b = delta
        nabla_w[-1] = activations[-2].T @ delta[-1]

        # move backward
        for i in range(2, self.total_layers):
            nabla_w[-i] = activations[-i-1].T @ delta[-i]

        return (nabla_w, nabla_b)
    
    def learn(self, batch, learning_rate = 3):
        """
            compute average nabla (nabla_w and nabla_b) for a batch of training data and 
            use that to update wieghts and biases.
            default learning rate is set to 3.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # array_backprop = lambda batch : [self.backprop(b[0], b[1]) for b in batch]
        # delta_w, delta_b = array_backprop(batch)
        # nabla_b = np.sum(delta_b)
        # nabla_w = np.sum(delta_w)
        
        for x, y in batch:
            # can i somehow parallelize this?
            # maybe use threads? or maybe cuda?
            delta_w, delta_b = self.backprop(x, y)
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
        
        self.weights = [w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb
                        for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, learning_rate, batch_size):
        """
            `training_data` should be an array of `(x, y)` tuples
            where `x` is the input vector and `y` is desired output vector
            `epochs` is the number of times to train
        """
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
    
    def evaluate_and_print(self, test_data):
        """
            for debug purposes
        """
        for x, y in test_data:
            pred = self.forwardpass(x)[0][-1]
            curr_cost = self.cost_func(pred, y)
            print(pred, y, curr_cost)
            print(np.argmax(pred), np.argmax(y))
            print(np.sum(pred))
