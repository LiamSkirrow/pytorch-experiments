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
            out_layer = mlp(from_numpy(image))
            loss = criterion(out_layer, from_numpy(target_output))
            loss.backward()
            optimiser.step()

def infer_on_samples(path, mlp):
    inference_dataset_images = path + "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
    inference_dataset_labels = path + "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

    correct_count = 0

    with open(inference_dataset_images, "rb") as image_file, \
         open(inference_dataset_labels, "rb") as label_file:
        # image header
        magic      = struct.unpack(">I", image_file.read(4))[0]
        num_images = struct.unpack(">I", image_file.read(4))[0]
        rows       = struct.unpack(">I", image_file.read(4))[0]
        cols       = struct.unpack(">I", image_file.read(4))[0]
        # label header
        label_magic = struct.unpack(">I", label_file.read(4))[0]
        num_labels  = struct.unpack(">I", label_file.read(4))[0]
        
        assert num_images == num_labels

        print('MNIST validation set details:')
        print('Magic number: ' + str(magic))
        print('Num Images: '   + str(num_images))

        # iterate over all existing images
        for i in range(0, num_images):
            # read the successive images, plot
            image_data = image_file.read(rows * cols)
            image = np.frombuffer(image_data, dtype=np.uint8)
            image_for_plot = image.reshape(rows, cols)

            # flatten + normalise the data (normalised 1D array, ready for input layer of MLP)
            image = image.reshape(rows * cols)
            image = image.astype(np.float32) / 255.0

            # infer the image through the NN
            output = mlp(from_numpy(image))

            # derive the label from the label data set
            # -> figure out the correct label for this image
            sample_label  = label_file.read(1)[0]

            # derive model output
            # output_layer = softmax(output)
            # model_estimate = int(np.argmax(output_layer))

            # manually check outputs
            sm = softmax(output[:10].detach().numpy())
            prediction = int(np.argmax(sm))
            # useful for debugging!
            # print('Sample number: ' + str(i) + ', model prediction:')
            # print(prediction)
            # plt.imshow(image_for_plot, cmap="gray")
            # plt.show()

            # check model correctness
            if(prediction == sample_label):
                correct_count += 1

            print('Validation dataset, image number: ', i, ', label: <', sample_label, '>')
            print('Model estimate: <', prediction, '>')
        
        model_correctness = (correct_count / float(num_images)) * 100.0
        return model_correctness

if __name__ == '__main__':
    # fetch the MNIST dataset from kagglehub
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    print("Path to dataset files:", path)

    # instantiate the multi-layer perceptron NN itself
    mlp = NeuralNet()

    print('Printing metadata of model')
    for name, parameter in mlp.named_parameters():
        print(name, parameter.shape)
    
    # begin the training sequence using the MNIST dataset
    print('Training model on training dataset...')
    train_on_samples(path, mlp)

    # infer the validation set through the NN, kepp track of accuracy score
    print('Running inference on validation dataset...')
    model_correctness = infer_on_samples(path, mlp)

    print('*** inference on validation set completed, final model correctness score: ***')
    print(str(model_correctness) + '\n')
