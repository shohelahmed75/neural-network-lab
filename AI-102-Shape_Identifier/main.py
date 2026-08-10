from network.network import NeuralNetwork
from services.prediction import predict, get_shape_name

def main():
    network_name = "Network_1"
    json_path = "output/network.json"

    # Sample input of 25 values (representing a 5x5 grid/image)
    sample_input = [
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0
    ]

    # Try prediction using saved network from output/network.json
    output = predict(network_name, sample_input, json_path)

    if output is not None:
        print(f"\n\033[96mExecuted prediction using saved network '{network_name}' from {json_path}\033[0m")
    else:
        print(f"\n\033[91mNetwork '{network_name}' not found in {json_path}. Creating new NeuralNetwork...\033[0m")
        net = NeuralNetwork()
        output = net.forward(sample_input)

    shape_name = get_shape_name(output)

    print(f"\nNetwork Output:\n{output}")
    print(f"\n\033[92mPredicted Shape: {shape_name.upper()}\033[0m")

if __name__ == "__main__":
    main()
