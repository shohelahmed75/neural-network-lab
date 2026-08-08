import math
import json
import os

# Hardcoded neuron name and number of training loops (epochs)
NEURON_NAME = "Neuron_3"
EPOCHS = 100000
LEARNING_RATE = 0.5


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def train():
    neuron_file = "output/neuron.json"
    data_file = "services/training/train_AND.json"

    # Check if neuron file exists
    if not os.path.exists(neuron_file):
        print("neuron not available")
        return

    # Load neurons data
    with open(neuron_file, "r") as f:
        try:
            neurons = json.load(f)
        except json.JSONDecodeError:
            neurons = {}

    # Check if specified neuron exists
    if NEURON_NAME not in neurons:
        print("neuron not available")
        return

    # Load dataset
    if not os.path.exists(data_file):
        print(f"Dataset file {data_file} not found.")
        return

    with open(data_file, "r") as f:
        dataset = json.load(f)

    # Get current neuron weights
    neuron = neurons[NEURON_NAME]
    w1 = neuron["w1"]
    w2 = neuron["w2"]
    bias = neuron["bias"]

    # Run training loop
    for epoch in range(1, EPOCHS + 1):
        for item in dataset:
            x1, x2, target = item[0], item[1], item[2]

            z = (x1 * w1) + (x2 * w2) + bias
            output = sigmoid(z)

            print(f"Iteration {epoch} [inputs: ({x1}, {x2})]: sigmoid(z) = {output}")

            # Gradient descent update for sigmoid neuron
            error = target - output
            delta = error * output * (1 - output)

            w1 += LEARNING_RATE * delta * x1
            w2 += LEARNING_RATE * delta * x2
            bias += LEARNING_RATE * delta

    # Update neuron data in output/neuron.json
    neuron["w1"] = w1
    neuron["w2"] = w2
    neuron["bias"] = bias
    neurons[NEURON_NAME] = neuron

    with open(neuron_file, "w") as f:
        json.dump(neurons, f, indent=4)

    print(f"Training completed for {NEURON_NAME}.")
    print(f"Updated weights for {NEURON_NAME}:")
    print(f"  w1: {w1}")
    print(f"  w2: {w2}")
    print(f"  bias: {bias}")


if __name__ == "__main__":
    train()