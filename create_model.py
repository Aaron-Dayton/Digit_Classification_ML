"""
    Author of network.py and create_model:
        Aaron Dayton  /  August 30, 2020
    Author of mnist_loader:
        Michal Dobrzanski
        Github Repo Link:
            https://github.com/MichalDanielDobrzanski/DeepLearningPython35

-------------------------------------------------------------------------------

    NOTE:   Aaron Dayton's network.py is modeled after Michal Dobrzanski's
            network.py code, but Aaron's feeds forward and performs the
            backpropagation algorithm on the entire mini batch at once, while
            Dobrzanski's feeds forward and performs the backpropagation
            algorithm on one instance in the mini batch at a time. This
            difference leads to a substantailly different implementation in most
            of the code, while reaching the same desired outcome.

-------------------------------------------------------------------------------

    Methods Used:

    mnist_loader.load_data_wrapper()
        Organizes the mnist data set into 3 distinct data sets, training_data
        (50000 instances), validation_data (10000 instances), test_data
        (10000 instances)

    network.Network():
        Takes the a list specifying the size of the neural network
        as a parameter. The list must have 784 at its first index and 10 at its
        last index. The list can be of any length >= 2 where all the indices
        store numbers > 0.

        Iinitializes a neural network with random weights and biases

    stochastic_gradient_descent:
        Takes the training data as the first parameter, the number of epochs
        as the second, the desired size of the mini batches used as the third,
        the learning rate as the fourth, and the data used for testing as the
        fifth (validataion_data can be inputed here when tuning the
        hyperparameters).

        Sets all the weights optimally throught the stochastic gradient descent
        algorithm

"""

#Initialize data

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

#Create network and perform stochastic gradient descent
import network

net = network.Network([784, 30, 10])
net.stochastic_gradient_descent(training_data, 30, 10, 3, test_data = test_data)
