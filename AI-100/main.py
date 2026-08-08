import json
from pathlib import Path
from models import NeuronV1
from services import predict_z, sigmoid

if __name__ == "__main__":

    neuron = NeuronV1(name="Neuron 1")

    print(f"\n\033[94mNeuron configuration: {neuron.json()}\033[0m")
    print("\n\033[96mPrediction:")

    predict_z(neuron, 1.0, 2.0)

    # Save neuron data to output/neuron.json
    output_path = Path(__file__).parent / "output" / "neuron.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(neuron.json(), f, indent=4)
    print(f"\n\033[92mSaved neuron data to {output_path}\033[0m")