import math

from src.models.neuron import NeuronV1


def predict_v1(neuron: NeuronV1, x1, x2):
    """
    Based on 2 given input numbers the trained/untrained neuron produces an output.

    :param neuron:
    :param x1:
    :param x2:
    :return: predicted value
    """
    _sum = (x1 * neuron.w1) + (x2 * neuron.w2) + neuron.bias
    return sigmoid(_sum)


def sigmoid(z):
    """
    Sigmoid squishes any number into a value between 0 and 1.
    :param z:
    :return:
    """
    return 1 / (1 + math.exp(-z))
