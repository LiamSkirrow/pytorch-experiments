import kagglehub
import matplotlib.pyplot as plt
# import struct
from scipy.special import softmax
from NeuralNet import NeuralNet

if __name__ == '__main__':
    # fetch the MNIST dataset from kagglehub
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)

    # instantiate the multi-layer perceptron NN itself
    mlp = NeuralNet()
    