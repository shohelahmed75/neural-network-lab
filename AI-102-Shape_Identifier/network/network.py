from network.layer import Layer

class NeuralNetwork:
    def __init__(self):
        self.hidden = Layer(25, 12)
        self.output = Layer(12, 3)

    def forward(self, inputs):
        hidden_output = self.hidden.forward(inputs)
        final_output = self.output.forward(hidden_output)

        return final_output