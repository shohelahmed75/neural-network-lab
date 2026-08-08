import math
import json
import os

# Hardcoded neuron name to predict with
NEURON_NAME = "Neuron_Failed_XOR"


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict():
    file_path = "output/neuron.json"

    # Check if file exists
    if not os.path.exists(file_path):
        print("neuron not available")
        return None

    # Load neurons data
    with open(file_path, "r") as f:
        try:
            neurons = json.load(f)
        except json.JSONDecodeError:
            neurons = {}

    # Check if specified neuron exists
    if NEURON_NAME not in neurons:
        print("neuron not available")
        return None

    # Get neuron parameters
    neuron = neurons[NEURON_NAME]
    w1 = neuron["w1"]
    w2 = neuron["w2"]
    bias = neuron["bias"]

    # Ask user for x1 and x2 inputs
    x1 = float(input("Enter x1: "))
    x2 = float(input("Enter x2: "))

    # Calculate z and sigmoid(z)
    z = (x1 * w1) + (x2 * w2) + bias
    result = sigmoid(z)

    # Save h value (sigmoid(z)) for this neuron
    neuron["h"] = result
    neurons[NEURON_NAME] = neuron
    with open(file_path, "w") as f:
        json.dump(neurons, f, indent=4)

    print(f"sigmoid(z): {result}")
    return result


if __name__ == "__main__":
    predict()