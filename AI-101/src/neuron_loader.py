import json
from typing import Any

from src.models.neuron import NeuronV1

NEURON_OUTPUT_PATH = "C:\\Users\\Mazid\\PycharmProjects\\AI-101\\output\\neuron.json"


def save_neuron(neuron: NeuronV1):
    all_neurons = load_all_neuron_data()

    # Put new/updated neuron
    all_neurons[neuron.name] = neuron.json()

    # Open file in write mode ('w') and save data
    with open(NEURON_OUTPUT_PATH, "w") as file_writer:
        json.dump(all_neurons, file_writer, indent=4)

def load_neuron_model(neuron_name: str) -> NeuronV1 | None:
    neuron_json = load_neuron_data(neuron_name)
    if not neuron_json:
        return None

    return NeuronV1(**neuron_json)

def load_neuron_data(neuron_name: str) -> dict[str, Any] | None:
    all_neurons = load_all_neuron_data()
    return all_neurons[neuron_name] if all_neurons else None


def load_all_neuron_data():
    with open(NEURON_OUTPUT_PATH, "r") as file:
        return json.load(file)
