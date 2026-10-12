import kagglehub
import matplotlib.pyplot as plt
import struct
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch import from_numpy
from scipy.special import softmax
from NeuralNet import NeuralNet

def train_on_samples(path, mlp):
    training_dataset_images = path + "/train-images-idx3-ubyte/train-images-idx3-ubyte"
    training_dataset_labels = path + "/train-labels-idx1-ubyte/train-labels-idx1-ubyte"

    # create the optimiser, define error criterion
    optimiser = optim.SGD(mlp.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    with open(training_dataset_images, "rb") as image_file, \
         open(training_dataset_labels, "rb") as label_file:
        # image header
        magic      = struct.unpack(">I", image_file.read(4))[0]
        num_images = struct.unpack(">I", image_file.read(4))[0]
        rows       = struct.unpack(">I", image_file.read(4))[0]
        cols       = struct.unpack(">I", image_file.read(4))[0]
        # label header
        label_magic = struct.unpack(">I", label_file.read(4))[0]
        num_labels  = struct.unpack(">I", label_file.read(4))[0]
        
        assert num_images == num_labels

        print('MNIST training set details:')
        print('Magic number: ' + str(magic))
        print('Num Images: '   + str(num_images))

        # enter training loop
        for i in range(0, num_images):
            # read the successive images, plot
            image_data = image_file.read(rows * cols)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image = image.reshape(rows, cols)
            # sanity checking plots
            # plt.imshow(image, cmap="gray")
            # plt.show()
            
            # flatten + normalise the data (normalised 1D array, ready for input layer of MLP)
            image = image.reshape(rows * cols)
            image = image.astype(np.float32) / 255.0

            # derive the target sample output
            sample_label  = label_file.read(1)[0]
            target_output = np.zeros(10, dtype=np.float32)     # TODO: can we find a way to avoid hardcoding this?
            target_output[sample_label] = 1

            # enter training loop... infer the flattened 1D image through the NN, calculate loss, backpropagate
            optimiser.zero_grad()
            out_layer = mlp.forward(from_numpy(image))
            loss = criterion(out_layer, from_numpy(target_output))
            loss.backward()
            optimiser.step()

if __name__ == '__main__':
    # fetch the MNIST dataset from kagglehub
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)

    # instantiate the multi-layer perceptron NN itself
    mlp = NeuralNet()
    
    # begin the training sequence using the MNIST dataset
    train_on_samples(path, mlp)
