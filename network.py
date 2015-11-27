import functools
import numpy
import iterutils
import mathutils

class Network:
    class Layer:
        def __init__(self, inputSize, layerSize):
            self.weights = numpy.random.uniform(-1.0, 1.0, (layerSize, inputSize))
            self.bias = numpy.random.uniform(-1.0, 1.0, (layerSize, 1))
            self.activationFunc = mathutils.sigmoid
            self.activationFuncDerivative = mathutils.sigmoid_prime

        def compute_output(self, input):
            return self.activationFunc(self.weights.dot(input) + self.bias)

        def output_layer_derror_by_doutput(self, expectation, outputs):
            return outputs[-1] - expectation

        def doutput_by_dactivation(self, layer_output):
            return self.activationFuncDerivative(layer_output)

        def dactivation_by_dweight(self, previous_layer_output):
            return previous_layer_output.transpose()

        def dactivation_by_dbias(self):
            return numpy.ones(self.bias.shape)

    def __init__(self, layerSizes):
        self.layers = [Network.Layer(inputSize, layerSize) for inputSize, layerSize in iterutils.window(layerSizes, 2)] 

    def compute_outputs(self, input):
        compute_and_accumulate_layer_outputs = (
            lambda prev_layer_outputs, layer: prev_layer_outputs + [layer.compute_output(prev_layer_outputs[-1])])
        return functools.reduce(compute_and_accumulate_layer_outputs, self.layers, [input])

    def compute_error(self, input, expectation):
        output = self.compute_outputs(input)[-1]
        return numpy.sum((expectation - output) ** 2) / expectation.shape[0]

    def compute_weight_and_bias_deltas(self, input, expectation, learning_rate):
        def calculate_derror_by_dactivation(layer, current_layer_output, derror_by_doutput):
            return layer.doutput_by_dactivation(current_layer_output) * derror_by_doutput

        def calculate_weight_and_bias_deltas(layer, previous_layer_output, derror_by_dactivation):
            dactivation_by_dweight = layer.dactivation_by_dweight(previous_layer_output)
            dactivation_by_dbias = layer.dactivation_by_dbias()

            weight_deltas = learning_rate * derror_by_dactivation * dactivation_by_dweight
            bias_deltas = learning_rate * derror_by_dactivation * dactivation_by_dbias
            return (weight_deltas, bias_deltas)

        outputs = self.compute_outputs(input)
        output_layer = self.layers[-1]

        output_layer_derror_by_doutput = output_layer.output_layer_derror_by_doutput(expectation, outputs)
        output_layer_derror_by_dactivation = calculate_derror_by_dactivation(output_layer, outputs[-1], output_layer_derror_by_doutput)
        
        weight_and_bias_deltas = [calculate_weight_and_bias_deltas(output_layer, outputs[-2], output_layer_derror_by_dactivation)]
        derror_by_dactivations = [output_layer_derror_by_dactivation]

        for index in reversed(range(len(self.layers[:-1]))):
            next_layer_derror_by_dactivation = derror_by_dactivations[0]
            dnext_layer_activation_by_doutput = self.layers[index + 1].weights.transpose()
            
            derror_by_doutput = dnext_layer_activation_by_doutput.dot(next_layer_derror_by_dactivation)
            derror_by_dactivation = calculate_derror_by_dactivation(self.layers[index], outputs[index + 1], derror_by_doutput)

            weight_and_bias_delta = calculate_weight_and_bias_deltas(self.layers[index], outputs[index], derror_by_dactivation)
            
            weight_and_bias_deltas = [weight_and_bias_delta] + weight_and_bias_deltas
            derror_by_dactivations = [derror_by_dactivation] + derror_by_dactivations

        return weight_and_bias_deltas

    def apply_weight_and_bias_deltas(self, weight_and_bias_deltas):
        for layer, (layer_weight_delta, layer_bias_delta) in zip(self.layers, weight_and_bias_deltas):
            layer.weights = layer.weights - layer_weight_delta
            layer.bias = layer.bias - layer_bias_delta
