import json
import os
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(network_name, inputs, json_path="output/network.json"):
    """
    Looks up network_name in json_path. If found, computes forward pass using the
    stored weights and biases, and returns the output. Otherwise, returns None.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        network_data = None

        if isinstance(data, dict):
            if network_name in data:
                network_data = data[network_name]
            elif data.get("name") == network_name:
                network_data = data
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("name") == network_name:
                    network_data = item
                    break

        if not network_data or "layers" not in network_data:
            return None

        current_inputs = inputs
        for layer in network_data["layers"]:
            layer_outputs = []
            for neuron in layer["neurons"]:
                weights = neuron["weights"]
                bias = neuron["bias"]
                
                total = bias
                for x, w in zip(current_inputs, weights):
                    total += x * w
                
                layer_outputs.append(sigmoid(total))
            current_inputs = layer_outputs

        return current_inputs

    except Exception as e:
        print(f"Warning: Failed to read prediction from {json_path}: {e}")
        return None

def get_shape_name(output_array):
    """
    Translates output neuron probabilities array into the predicted shape name.
    Index 0: square
    Index 1: cross
    Index 2: triangle
    """
    if not output_array or len(output_array) < 3:
        return "unknown"
    shapes = ["square", "cross", "triangle"]
    max_idx = output_array.index(max(output_array))
    return shapes[max_idx]
