from django.http import HttpResponse
import numpy
import scipy.special
import os
import scipy.ndimage

# Read Data
pwd = os.path.dirname(__file__)

learn_data_file = open(os.path.join(pwd, 'mnist', '60000.csv'), 'r')
learn_data_list = learn_data_file.readlines()
learn_data_file.close()

test_data_file = open(os.path.join(pwd, 'mnist', '10000.csv'), 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningRate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # Query
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Learn

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
             numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
             numpy.transpose(inputs))

        pass
        
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        # query

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.01

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

epochs = 10

for e in range(epochs):
    for learnItem in learn_data_list:
        # Normalize data
        all_values = learnItem.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        nn.train(inputs, targets)

        # Rotation
        inputs_plus_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        nn.train(inputs_plus_img.reshape(784), targets)

        inputs_minus_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        nn.train(inputs_minus_img.reshape(784), targets)

        pass
    pass

def index(request):
    
    score = 0
    for testItem in test_data_list:
        all_values = testItem.split(',')
        expect = all_values[0]
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        result = numpy.argmax(nn.query(inputs))
        print('result: ', result)
        print('expect: ', expect)
        print(' ')
        if result == int(expect): score = score+1

        pass

    print('score: ', score)
    #print(nn.query([1.2, 4.2, 42.2]))
    return HttpResponse("Hello, world. You're at the polls index.")