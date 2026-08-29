from models.neuronV2 import Neuron

class Layer:
    def __init__(self, input_count, neuron_count, name=None):
        self.name = name
        self.neurons = [
            Neuron(input_count)
            for _ in range(neuron_count)
        ]

    def forward(self, inputs):
        return [
            neuron.forward(inputs)
            for neuron in self.neurons
        ]

    def to_dict(self):
        layer_data = {
            "neurons": [neuron.to_dict() for neuron in self.neurons]
        }
        if self.name:
            layer_data["name"] = self.name
        return layer_data