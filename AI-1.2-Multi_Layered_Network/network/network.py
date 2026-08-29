import json
import os
from network.layer import Layer

class NeuralNetwork:
    def __init__(self, name="Network_1"):
        self.name = name
        self.hidden = Layer(25, 12, name="hidden")
        self.output = Layer(12, 3, name="output")
        self.layers = [self.hidden, self.output]

    def forward(self, inputs):
        hidden_output = self.hidden.forward(inputs)
        final_output = self.output.forward(hidden_output)

        return final_output

    def to_dict(self):
        return {
            "name": self.name,
            "layers": [layer.to_dict() for layer in self.layers]
        }

    def save_to_json(self, filepath="output/network.json"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)