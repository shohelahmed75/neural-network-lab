import json
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.network import NeuralNetwork
from models.neuronV2 import sigmoid

def train_network(network, dataset, epochs, learning_rate):
    """
    Trains the neural network using backpropagation over the dataset.
    """
    print(f"Starting training for {epochs} epochs with learning rate {learning_rate}...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for sample in dataset["shapes"]:
            x = sample["data"]
            target = sample["target"]

            # Forward pass
            h_outputs = network.hidden.forward(x)
            final_outputs = network.output.forward(h_outputs)

            # Calculate Loss (MSE)
            loss = sum(0.5 * (out - t) ** 2 for out, t in zip(final_outputs, target))
            total_loss += loss

            # 1. Calculate Output Layer deltas: delta_out = (output - target) * output * (1 - output)
            out_deltas = []
            for out, t in zip(final_outputs, target):
                d_out = (out - t) * out * (1 - out)
                out_deltas.append(d_out)

            # 2. Calculate Hidden Layer deltas: delta_h = sum(delta_out * w_out) * h * (1 - h)
            hidden_deltas = []
            for j, h_val in enumerate(h_outputs):
                sum_err = sum(out_deltas[i] * network.output.neurons[i].weights[j] for i in range(len(final_outputs)))
                d_h = sum_err * h_val * (1 - h_val)
                hidden_deltas.append(d_h)

            # 3. Update Output Layer weights and biases
            for i, neuron in enumerate(network.output.neurons):
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learning_rate * out_deltas[i] * h_outputs[j]
                neuron.bias -= learning_rate * out_deltas[i]

            # 4. Update Hidden Layer weights and biases
            for j, neuron in enumerate(network.hidden.neurons):
                for k in range(len(neuron.weights)):
                    neuron.weights[k] -= learning_rate * hidden_deltas[j] * x[k]
                neuron.bias -= learning_rate * hidden_deltas[j]

        if epoch == 1 or epoch % 1000 == 0 or epoch == epochs:
            avg_loss = total_loss / len(dataset["shapes"])
            print(f"Epoch {epoch:5d}/{epochs} - Loss: {avg_loss:.6f}")

    print("Training complete!")

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "training_shapes.json")
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "network.json")

    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Initialize neural network
    net = NeuralNetwork(name="Network_1")

    # Train network for 5000 epochs
    train_network(net, dataset, epochs=1000000, learning_rate=0.5)

    # Save trained parameters (weights, bias, h) to output/network.json
    net.save_to_json(output_path)
    print(f"\nTrained network parameters saved successfully to {output_path}")

if __name__ == "__main__":
    main()
