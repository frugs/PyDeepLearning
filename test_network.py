import unittest
import random
import numpy.testing as npt
from network import *


class TestNetwork(unittest.TestCase):
    def test_2_2_2_network(self):
        network = Network([2, 2, 2])
        network.layers[0].weights = numpy.array([[0.15, 0.20],
                                                 [0.25, 0.30]])
        network.layers[0].bias = numpy.array([[0.35], [0.35]])
        network.layers[1].weights = numpy.array([[0.40, 0.45],
                                                 [0.50, 0.55]])
        network.layers[1].bias = numpy.array([[0.60], [0.60]])

        learning_rate = 0.5

        training_input = numpy.array([[0.05], [0.1]])
        training_target = numpy.array([[0.01], [0.99]])
        outputs = network.compute_outputs(training_input)
        expected_output = numpy.array([[0.75136507], [0.772928465]])

        npt.assert_almost_equal(outputs[-1], expected_output)

        weight_and_bias_deltas = network.compute_weight_and_bias_deltas(training_input, training_target, learning_rate)
        network.apply_weight_and_bias_deltas(weight_and_bias_deltas)

        expected_weights = [numpy.array([[0.149780716, 0.19956143],
                                         [0.24975114, 0.29950229]]),
                            numpy.array([[0.35891648, 0.408666186],
                                         [0.511301270, 0.561370121]])]

        npt.assert_almost_equal(network.layers[0].weights, expected_weights[0])
        npt.assert_almost_equal(network.layers[1].weights, expected_weights[1])

    def test_2_2_1_network_learning(self):
        network = Network([2, 2, 1])
        network.layers[0].weights = numpy.array([[0.1, 0.8],
                                                 [0.4, 0.6]])
        network.layers[0].bias = numpy.array([[0], [0]])
        network.layers[1].weights = numpy.array([[0.3, 0.9]])
        network.layers[1].bias = numpy.array([[0]])

        learning_rate = 0.5

        training_input = numpy.array([[0.35], [0.9]])
        training_target = numpy.array([[0.5]])

        for _ in range(10000):
            weight_and_bias_deltas = network.compute_weight_and_bias_deltas(training_input, training_target,
                                                                            learning_rate)
            network.apply_weight_and_bias_deltas(weight_and_bias_deltas)

        error = network.compute_error(training_input, training_target)
        npt.assert_array_less(error, 0.05)

    def test_2_2_1_network_xor(self):
        network = Network([2, 2, 1])
        network.layers[0].weights = numpy.array([[0.129952, -0.923123],
                                                 [0.570345, -0.328932]])
        network.layers[0].bias = numpy.array([[0.341232], [-0.115234]])
        network.layers[1].weights = numpy.array([[0.164732, 0.752621]])
        network.layers[1].bias = numpy.array([[-0.993423]])

        learning_rate = 0.5

        training_input = numpy.array([[0], [0]])
        training_target = numpy.array([[0]])

        expected_output = numpy.array([[0.367610]])
        outputs = network.compute_outputs(training_input)

        npt.assert_almost_equal(outputs[-1], expected_output, 4)

        expected_weight_deltas = [numpy.array([[0, 0],
                                               [0, 0]]),
                                  numpy.array([[0.024975, 0.020135]])]

        expected_bias_deltas = [numpy.array([[0.0017095],
                                             [0.0080132]]),
                                numpy.array([[0.042730]])]

        weight_deltas, bias_deltas = zip(*network.compute_weight_and_bias_deltas(training_input, training_target, learning_rate))

        for weight_delta, expected_weight_delta in zip(weight_deltas, expected_weight_deltas):
            npt.assert_almost_equal(weight_delta, expected_weight_delta, 5)

        for bias_delta, expected_bias_delta in zip(bias_deltas, expected_bias_deltas):
            npt.assert_almost_equal(bias_delta, expected_bias_delta, 5)

    def test_xor_function_learning_with_2_2_1_network(self):
        network = Network([2, 2, 1])
        network.layers[0].weights = numpy.array([[0.129952, -0.923123],
                                                 [0.570345, -0.328932]])
        network.layers[0].bias = numpy.array([[0.341232], [-0.115234]])
        network.layers[1].weights = numpy.array([[0.164732, 0.752621]])
        network.layers[1].bias = numpy.array([[-0.993423]])

        learning_rate = 0.5

        training_inputs = [numpy.array([[1], [1]]),
                           numpy.array([[1], [0]]),
                           numpy.array([[0], [1]]),
                           numpy.array([[0], [0]])]

        training_targets = [numpy.array([[0]]),
                            numpy.array([[1]]),
                            numpy.array([[1]]),
                            numpy.array([[0]])]

        training_set = list(zip(training_inputs, training_targets))

        for _ in range(50000):
            training_input, training_target = training_set[random.randrange(0, len(training_inputs))]

            pre_training_error = network.compute_error(training_input, training_target)

            weight_and_bias_deltas = network.compute_weight_and_bias_deltas(training_input, training_target,
                                                                            learning_rate)
            network.apply_weight_and_bias_deltas(weight_and_bias_deltas)

            post_training_error = network.compute_error(training_input, training_target)

            npt.assert_array_less(post_training_error, pre_training_error)

        for training_input, training_target in zip(training_inputs, training_targets):
            error = network.compute_error(training_input, training_target)
            npt.assert_array_less(error, 0.05)
