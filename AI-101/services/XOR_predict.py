import math
import json
import os


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def predict():
    file_path = "output/neuron.json"

    if not os.path.exists(file_path):
        print("Connected Neuron output not available")
        return None

    with open(file_path, "r") as f:
        try:
            neurons = json.load(f)
        except json.JSONDecodeError:
            neurons = {}

    # Get Neuron_1 and Neuron_2 outputs
    n1 = neurons.get("Neuron_1", {})
    n2 = neurons.get("Neuron_2", {})

    h1 = n1.get("h", "")
    h2 = n2.get("h", "")

    # Check if h values are valid numbers
    if not isinstance(h1, (int, float)) or not isinstance(h2, (int, float)):
        print("Connected Neuron output not available")
        return None

    x1 = float(h1)
    x2 = float(h2)

    # Get XOR neuron (Neuron_XOR) parameters
    neuron = neurons.get("Neuron_XOR", {})
    w1 = neuron.get("w1", 0.5)
    w2 = neuron.get("w2", 0.5)
    bias = neuron.get("bias", -0.5)

    # Predict
    z = (x1 * w1) + (x2 * w2) + bias
    result = sigmoid(z)

    # Store h value for Neuron_XOR
    if "Neuron_XOR" in neurons:
        neurons["Neuron_XOR"]["h"] = result
        with open(file_path, "w") as f:
            json.dump(neurons, f, indent=4)

    print(f"sigmoid(z): {result}")
    return result


if __name__ == "__main__":
    predict()
