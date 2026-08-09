from network.network import NeuralNetwork

def main():
    # Instantiate the neural network (25 inputs -> 12 hidden -> 3 outputs)
    net = NeuralNetwork()

    # Sample input of 25 values (representing a 5x5 grid/image)
    sample_input = [
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1
    ]

    # Run forward pass
    output = net.forward(sample_input)

    print("Network Output:")
    for i, score in enumerate(output):
        print(f"  Output Neuron {i + 1}: {score:.4f}")

if __name__ == "__main__":
    main()
