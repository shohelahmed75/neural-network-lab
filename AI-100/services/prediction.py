import math
import sys
from pathlib import Path

# Ensure root directory is in sys.path for top-level package imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.neuron import NeuronV1


def predict_z(neuron: NeuronV1, x1, x2):
    z = (x1 * neuron.w1) + (x2 * neuron.w2) + neuron.bias
    res = sigmoid(z)
    print(res)
    return res


def sigmoid(z):
    return 1 / (1 + math.exp(-z))