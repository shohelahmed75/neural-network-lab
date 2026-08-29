import random
import math

def sigmoid(z):
        return 1 / (1 + math.exp(-z))

class Neuron:
    def __init__(self, input_count):
        self.weights = [
            random.uniform(-1, 1)
            for _ in range(input_count)
        ]

        self.bias = random.uniform(-1, 1)
        self.h = None

    def forward(self, inputs):
        total = self.bias

        for x, w in zip(inputs, self.weights):
            total += x * w

        self.h = sigmoid(total)
        return self.h

    def to_dict(self):
        return {
            "weights": self.weights,
            "bias": self.bias,
            "h": self.h
        }