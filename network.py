import random
import numpy as np


class Network(object):

    def __init__(self, sizes):
        '''This code initializes the network object.
        The weights and biases will be initialized
        using a Gaussian distribution with a mean of
        0 and a standard deviation of 1.'''
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        '''Preforms the stochastic gradient descent algorithm to update the
        biases and weights in the network to allow for better preformance.
        First, the training and test data are put into the correct formats,
        then the training data is shuffled and arranged into mini-batches.
        Those mini-batches are passed to 'update_weights_and_biases' which does
        most of the work. Finally, if test data is included, the number of
        correctly classified digits are counted'''
        training_data = list(training_data)
        training_data_length = len(training_data)
        if test_data:
            test_data = list(test_data)
            test_data_length = len(test_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for n in range(0, training_data_length, mini_batch_size):
                mini_batch = [instance for instance in training_data[n:n + mini_batch_size]]
                mini_batches.append(mini_batch)
            for mini_batch in mini_batches:
                self.update_weights_and_biases(learning_rate, mini_batch_size, mini_batch)
            if test_data:
                print("Epoch ", e, " : ", self.evaluate(test_data), " / ", test_data_length)
            else:
                print("Epoch ", e, ": Complete")


    def update_weights_and_biases(self, learning_rate, mini_batch_size, mini_batch):
        '''Changes the biases and weights by nabla_b and nable_w respectively,
        which are calculated in the backpropagation definition.'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b, nabla_w = self.backpropagation(mini_batch)
        self.biases = [np.add(b, -(learning_rate/mini_batch_size)*nb) for b, nb in zip(self.biases, nabla_b)]
        self.weights = [np.add(w, -(learning_rate/mini_batch_size)*nw) for w, nw in zip(self.weights, nabla_w)]

    def backpropagation(self, mini_batch):
        '''Returns the tupple (nabla_b, nabla_w) that the weights and biases
        will be changed by respectively. This tupple is the gradient of
        the cost function.'''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        layer_one_activations = [] # list to store all the first layer activations from each instance
        correct_final_layer_outputs = [] # list to store all the final layer outputs from each instance
        for x, y in mini_batch:
            layer_one_activations.append(x)
            correct_final_layer_outputs.append(y)
        all_activations = [layer_one_activations] # list to store all activations in the network
        curr_activations = layer_one_activations
        weighted_inputs = [] # list to store all the weighted inputs
        # feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, curr_activations)
            for neurons in range(0, len(z)):
                z[neurons] = np.add(z[neurons], b)
            weighted_inputs.append(z)
            curr_activations = self.sigmoid(z)
            all_activations.append(curr_activations)
        # backward pass, output layer calculation
        delta = self.quadratic_cost_derivative(all_activations[-1], correct_final_layer_outputs) * self.sigmoid_derivative(all_activations[-1])
        nabla_b_place_holder = delta
        nabla_w_place_holder = [np.matmul(e, np.transpose(a)) for e, a in zip(delta, all_activations[-2])]
        for n in range(0, len(nabla_b_place_holder)):
            nabla_b[-1] = np.add(nabla_b[-1], nabla_b_place_holder[n])
            nabla_w[-1] = np.add(nabla_w[-1], nabla_w_place_holder[n])
        # backwards pass, all other layers
        for l in range(2, self.num_layers):
            z = weighted_inputs[-l]
            sd = self.sigmoid_derivative(z)
            delta = np.matmul(self.weights[-l+1].transpose(), delta) * sd
            nabla_b_place_holder = delta
            nabla_w_place_holder = [np.matmul(e, np.transpose(a)) for e, a in zip(delta, all_activations[-l-1])]
            for n in range(0, len(nabla_b_place_holder)):
                nabla_b[-l] = np.add(nabla_b[-l], nabla_b_place_holder[n])
                nabla_w[-l] = np.add(nabla_w[-l], nabla_w_place_holder[n])
        return (nabla_b, nabla_w)

    def quadratic_cost_derivative(self, activations, correct_final_layer_outputs):
        '''Returns the partial derivative of the quadratic cost function.'''
        return activations-correct_final_layer_outputs

    def sigmoid(self, weighted_inputs):
        return 1.0/(1.0+np.exp(-weighted_inputs))

    def sigmoid_derivative(self, weighted_inputs):
        return self.sigmoid(weighted_inputs)*(1.0-self.sigmoid(weighted_inputs))

    def evaluate(self, test_data):
        '''Returns the total number of correctly classified instances of test
        data. First, feed the input data, x, forward through the network, then
        put those outputs together in a list with the correct outputs, y,
        finally, count how many of the network's classifictions are correct.'''
        curr_activations = []
        correct_final_layer_outputs = []
        for x, y in test_data:
            curr_activations.append(x)
            correct_final_layer_outputs.append(y)
        # feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.matmul(w, curr_activations)
            for neuron in range(0, len(z)):
                z[neuron] = np.add(z[neuron], b)
            curr_activations = self.sigmoid(z)
        # add up number of correct claasifications
        results = []
        for n in range(len(curr_activations)):
            results.append((np.argmax(curr_activations[n]), correct_final_layer_outputs[n]))
        return sum(int(x == y) for x, y in results)
