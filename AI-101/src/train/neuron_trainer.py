import json
import random
from pathlib import Path

from src.models.neuron import NeuronV1
from src.neuron_loader import save_neuron
from src.services.prediction_service import predict_v1

BASE_DIR = Path(__file__).resolve().parent


class NeuronTrainer:
    training_data = []
    training_data_path = BASE_DIR / "training_data.json"

    learning_rate = 0.5  # how big a nudge we make each time
    epochs = 20000  # how many times we loop over the data

    # Logging properties
    do_log = True
    log_after_train = 1000

    def __init__(self, name, training_data_path=None):
        """
        Initialize the Neuron Trainer class instance with two weights and one bias.
        Start them random - the neuron knows nothing yet.
        """
        self.neuron = NeuronV1(
            name=name,
            w1=random.uniform(-1, 1),
            w2=random.uniform(-1, 1),
            bias=random.uniform(-1, 1),
        )

        if training_data_path:
            self.training_data_path = Path(training_data_path)

        # Load training data
        self.__load_train_data__()

    def __load_train_data__(self):
        with open(self.training_data_path, "r") as file:
            self.training_data = json.load(file)

    def start_training(self, n: int):
        for i in range(n):
            self.__train__()

            # Every so often, print how the neuron is doing.
            if self.do_log and i % self.log_after_train == 0:
                total_error = sum(
                    (predict_v1(self.neuron, a, b) - t) ** 2 for a, b, t in self.training_data
                )
                print(f"after {i:5d} training runs   error {total_error:.8f}")

        print(f"\n====Training finished for neuron '{self.neuron.name}' ====")
        print("Final learned weights:")
        print(f"  w1 = {self.neuron.w1:.3f}   w2 = {self.neuron.w2:.3f}   bias = {self.neuron.bias:.3f}\n")

    def save(self):
        print("Saving neuron")
        save_neuron(self.neuron)

    def __train__(self):
        """
        Train the Neuron once with the training data
        """
        for x1, x2, target in self.training_data:
            output = predict_v1(self.neuron, x1, x2)  # the guess
            error = output - target

            # delta = error scaled by how much a weight change actually
            # moves the output right here (the sigmoid slope).
            delta = error * self.__sigmoid_slope__(output)

            # Nudge each weight in the direction that reduces error.
            # A weight only matters as much as its input was "on".
            self.neuron.w1 -= self.learning_rate * delta * x1
            self.neuron.w2 -= self.learning_rate * delta * x2
            self.neuron.bias -= self.learning_rate * delta  # bias input is always 1

    def __sigmoid_slope__(self, _output):
        # How steep the sigmoid is at this point. Needed for learning.
        # Handy fact: the slope can be written using the output itself.
        return _output * (1 - _output)
