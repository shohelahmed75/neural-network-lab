import random


class NeuronV1:
    def __init__(self, name, w1=None, w2=None, bias=None):
        self.name = name
        self.w1 = random.uniform(-1.0, 1.0) if w1 is None else w1
        self.w2 = random.uniform(-1.0, 1.0) if w2 is None else w2
        self.bias = random.uniform(-1.0, 1.0) if bias is None else bias

    def json(self) -> dict:
        return {
            "name": self.name,
            "w1": self.w1,
            "w2": self.w2,
            "bias": self.bias
        }
