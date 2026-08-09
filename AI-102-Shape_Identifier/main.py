from network.network import NeuralNetwork
from services.prediction import predict

def main():
    network_name = "Network_1"
    json_path = "output/network.json"

    # Sample input of 25 values (representing a 5x5 grid/image)
    sample_input = [
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ]

    # Try prediction using saved network from output/network.json
    output = predict(network_name, sample_input, json_path)

    if output is not None:
        print(f"\n\033[96mExecuted prediction using saved network '{network_name}' from {json_path}")
    else:
        print(f"\n\033[91mNetwork '{network_name}' not found in {json_path}. Executing old usual way (new NeuralNetwork)...")
        net = NeuralNetwork()
        output = net.forward(sample_input)

    print("\nNetwork Output:")
    for i, score in enumerate(output):
        print(f"  Output Neuron {i + 1}: {score:.4f}")

if __name__ == "__main__":
    main()
