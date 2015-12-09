import numpy


def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_prime(y):
    return y * (1 - y)