import math
import json
import os


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

    # Check if required neurons exist
    if "Neuron_1" not in neurons or "Neuron_2" not in neurons or "Neuron_XOR" not in neurons:
        print("neuron not available")
        return None

    # Ask user for x1 and x2 inputs
    x1 = float(input("Enter x1: "))
    x2 = float(input("Enter x2: "))

    # Feed x1 and x2 into Neuron_1 -> calculate h1
    n1 = neurons["Neuron_1"]
    z1 = (x1 * n1["w1"]) + (x2 * n1["w2"]) + n1["bias"]
    h1 = sigmoid(z1)
    n1["h"] = h1

    # Feed x1 and x2 into Neuron_2 -> calculate h2
    n2 = neurons["Neuron_2"]
    z2 = (x1 * n2["w1"]) + (x2 * n2["w2"]) + n2["bias"]
    h2 = sigmoid(z2)
    n2["h"] = h2

    # Feed h1 and h2 as inputs into Neuron_XOR -> calculate result
    nx = neurons["Neuron_XOR"]
    z_xor = (h1 * nx["w1"]) + (h2 * nx["w2"]) + nx["bias"]
    result = sigmoid(z_xor)
    nx["h"] = result

    # Save updated h values to output/neuron.json
    neurons["Neuron_1"] = n1
    neurons["Neuron_2"] = n2
    neurons["Neuron_XOR"] = nx
    with open(file_path, "w") as f:
        json.dump(neurons, f, indent=4)

    print(f"sigmoid(z): {result}")
    return result


if __name__ == "__main__":
    predict()
