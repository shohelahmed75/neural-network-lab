class NeuronV1:
    def __init__(self, name, w1, w2, bias):
        self.name = name
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    def json(self) -> dict:
        return {
            "name": self.name,
            "w1": self.w1,
            "w2": self.w2,
            "bias": self.bias
        }
