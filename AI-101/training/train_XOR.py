import math
import json
import os

EPOCHS = 100000
LEARNING_RATE = 0.5


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def train():
    neuron_file = "output/neuron.json"
    data_file = "training/train_XOR.json"

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

    # Check if required neurons exist
    if "Neuron_1" not in neurons or "Neuron_2" not in neurons or "Neuron_XOR" not in neurons:
        print("neuron not available")
        return

    # Load XOR dataset
    if not os.path.exists(data_file):
        print(f"Dataset file {data_file} not found.")
        return

    with open(data_file, "r") as f:
        dataset = json.load(f)

    # Get weights and bias for Neuron_1
    n1 = neurons["Neuron_1"]
    w1_n1, w2_n1, bias_n1 = n1["w1"], n1["w2"], n1["bias"]

    # Get weights and bias for Neuron_2
    n2 = neurons["Neuron_2"]
    w1_n2, w2_n2, bias_n2 = n2["w1"], n2["w2"], n2["bias"]

    # Get weights and bias for Neuron_XOR
    nx = neurons["Neuron_XOR"]
    w1_xor, w2_xor, bias_xor = nx["w1"], nx["w2"], nx["bias"]

    # Run training loop
    for epoch in range(1, EPOCHS + 1):
        for item in dataset:
            x1, x2, target = item[0], item[1], item[2]

            # Step 1: Forward pass for Neuron_1 and Neuron_2
            z1 = (x1 * w1_n1) + (x2 * w2_n1) + bias_n1
            h1 = sigmoid(z1)

            z2 = (x1 * w1_n2) + (x2 * w2_n2) + bias_n2
            h2 = sigmoid(z2)

            # Step 2: Forward pass for Neuron_XOR using h1 and h2 as inputs
            z_xor = (h1 * w1_xor) + (h2 * w2_xor) + bias_xor
            output = sigmoid(z_xor)

            # Step 3: Find error (result - target)
            error = output - target

            # Step 4: Find Delta_XOR
            delta_xor = error * output * (1 - output)

            # Step 6: Backpropagation - find delta for Neuron_1 and Neuron_2
            delta_n1 = delta_xor * w1_xor * h1 * (1 - h1)
            delta_n2 = delta_xor * w2_xor * h2 * (1 - h2)

            # Step 5: Update weights and bias of Neuron_XOR
            w1_xor -= LEARNING_RATE * delta_xor * h1
            w2_xor -= LEARNING_RATE * delta_xor * h2
            bias_xor -= LEARNING_RATE * delta_xor

            # Step 7: Update weights and bias of Neuron_1 and Neuron_2
            w1_n1 -= LEARNING_RATE * delta_n1 * x1
            w2_n1 -= LEARNING_RATE * delta_n1 * x2
            bias_n1 -= LEARNING_RATE * delta_n1

            w1_n2 -= LEARNING_RATE * delta_n2 * x1
            w2_n2 -= LEARNING_RATE * delta_n2 * x2
            bias_n2 -= LEARNING_RATE * delta_n2

    # Save updated weights and bias to output/neuron.json
    neurons["Neuron_1"]["w1"] = w1_n1
    neurons["Neuron_1"]["w2"] = w2_n1
    neurons["Neuron_1"]["bias"] = bias_n1

    neurons["Neuron_2"]["w1"] = w1_n2
    neurons["Neuron_2"]["w2"] = w2_n2
    neurons["Neuron_2"]["bias"] = bias_n2

    neurons["Neuron_XOR"]["w1"] = w1_xor
    neurons["Neuron_XOR"]["w2"] = w2_xor
    neurons["Neuron_XOR"]["bias"] = bias_xor

    with open(neuron_file, "w") as f:
        json.dump(neurons, f, indent=4)

    print("Training completed for XOR network.")
    print("Updated weights for Neuron_1:", neurons["Neuron_1"])
    print("Updated weights for Neuron_2:", neurons["Neuron_2"])
    print("Updated weights for Neuron_XOR:", neurons["Neuron_XOR"])


if __name__ == "__main__":
    train()
