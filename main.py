import numpy as np
import neuralnet as net
import mnist_loader as loader

mnist = loader.MnistDataloader(
    loader.training_img_path, 
    loader.training_label_path,
    loader.test_img_path,
    loader.test_label_path,
)

training_data_, test_data_ = mnist.load_data()
training_data = list(zip(training_data_[0], training_data_[1]))
test_data = list(zip(test_data_[0], test_data_[1]))

mnist_net = net.NeuralNet([28*28, 15, 10])
mnist_net.SGD(training_data, 30, 3, 10)
mnist_net.save_model("mnist_model_001.pkl")