import random
import json
import os


class NeuronV1:
    def __init__(self, name, w1=None, w2=None, bias=None):
        self.name = name
        self.w1 = random.uniform(-1.0, 1.0) if w1 is None else w1
        self.w2 = random.uniform(-1.0, 1.0) if w2 is None else w2
        self.bias = random.uniform(-1.0, 1.0) if bias is None else bias

    def json(self) -> dict:
        return {
            "name": self.name,
            "w1": self.w1,
            "w2": self.w2,
            "bias": self.bias,
            "h": ""
        }


if __name__ == "__main__":
    file_path = "output/neuron.json"

    # Load existing neurons if file exists
    neurons = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                neurons = json.load(f)
            except json.JSONDecodeError:
                neurons = {}

    # Generate serial number (Neuron_1, Neuron_2, etc.)
    serial_number = len(neurons) + 1
    neuron_name = f"Neuron_{serial_number}"

    # Create new neuron with random values
    new_neuron = NeuronV1(name=neuron_name)
    neurons[neuron_name] = new_neuron.json()

    # Save updated neurons to output/neuron.json
    os.makedirs("output", exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(neurons, f, indent=4)

    print(f"Created {neuron_name} and saved to {file_path}")
    print(neurons[neuron_name])
