import numpy as np
import neuralnet as net
import mnist_loader as loader

mnist = loader.MnistDataloader(
    loader.training_img_path, 
    loader.training_label_path,
    loader.test_img_path,
    loader.test_label_path,
)

training_data, test_data = mnist.load_data()

mnist_net = net.NeuralNet([28*28, 15, 10])

def train(modelname : str):
    mnist_net.SGD(training_data, 300, 1.5, 100)
    mnist_net.save_model(modelname)

def test(modelname : str):
    mnist_net.load_model(modelname)
    mnist_net.evaluate(test_data)
