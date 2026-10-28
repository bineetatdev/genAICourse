import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(input_size, output_size):
    return np.random.rand(input_size, output_size)




def numpy_vector_multiply(inputs, weights, bias):
     #Convert lists to numpy arrays
    inputs_array = np.array(inputs)
    weights_array = np.array(weights)

    # Calculate the dot product and add the bias
    outputs = np.dot(weights_array, inputs_array) + bias

    return outputs

def vector_multiply(inputs, weights, bias):
    output = (inputs[0] * weights[0] +
    inputs[1]*weights[1] +
    inputs[2]*weights[2] +
    inputs[3]*weights[3] + bias)
    return output


def numpy_vector_matrix_multiply(inputs, weights, biases):

    # Convert lists to numpy arrays
    inputs_array = np.array(inputs)
    weights_array = np.array(weights)
    biases_array = np.array(biases)

    # Calculate the dot product and add the biases
    layer_outputs = np.dot(weights_array, inputs_array) + biases_array
    return layer_outputs

def vector_matrix_multiply(inputs, weights, biases):

    # Output of current layer
    layer_outputs = []

    # For each neuron
    for neuron_weights, neuron_bias in zip(weights, biases):
        # Zeroed output of given neuron
        neuron_output = 0
        # For each input and weight to the neuron
        for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron's output variable
            neuron_output += n_input*weight ## W31*X1 + W32*X2 + W33*X3 + W34*X4
        # Add bias
        neuron_output += neuron_bias ## ## W31*X1 + W32*X2 + W33*X3 + W34*X4 + B3
        # Put neuron's result to the layer's output list
        layer_outputs.append(neuron_output)
    print(layer_outputs)


class dense_layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(output_size)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases


    #input_size = 3
    #output_size = 2
    #layer1 = dense_layer(input_size, output_size)
inputs = [1.0, 2.0, 4.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
result = numpy_vector_multiply(inputs, weights, bias)
print("Result of vector multiplication:", result)

inputs1 = [1, 2, 3, 2.5]

##LIST OF WEIGHTS
weights1 = [[0.2, 0.8, -0.5, 1],
[0.5, -0.91, 0.26, -0.5],
[-0.26, -0.27, 0.17, 0.87]]

##LIST OF BIASES
biases = [2, 3, 0.5]

result = numpy_vector_matrix_multiply(inputs1, weights1, biases)
print("Result of vector matrix multiplication:", result)

