from models.neuronV2 import Neuron

class Layer:
    def __init__(self, input_count, neuron_count):
        self.neurons = [
            Neuron(input_count)
            for _ in range(neuron_count)
        ]

    def forward(self, inputs):
        return [
            neuron.forward(inputs)
            for neuron in self.neurons
        ]