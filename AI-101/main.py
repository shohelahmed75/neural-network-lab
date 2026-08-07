"""
A single neuron (perceptron) built completely from scratch.
No libraries doing the thinking for us — just plain Python + math.exp.

Goal: teach the neuron the logical AND function.
    inputs (0,0) -> 0
    inputs (0,1) -> 0
    inputs (1,0) -> 0
    inputs (1,1) -> 1

The whole point is the LEARNING LOOP:
    guess -> measure error -> nudge weights -> repeat.
That exact loop is what scales all the way up to a chatbot.
"""

import random
from pathlib import Path

from src.models.neuron import NeuronV1
from src.neuron_loader import save_neuron
from src.services.prediction_service import predict_v1
from src.train.neuron_trainer import NeuronTrainer

# def sigmoid_slope(output):
#     # How steep the sigmoid is at this point. Needed for learning.
#     # Handy fact: the slope can be written using the output itself.
#     return output * (1 - output)
#
#
# # ---------------------------------------------------------------
# # 2. The neuron's "brain": two weights and one bias.
# #    Start them random — the neuron knows nothing yet.
# # ---------------------------------------------------------------
# w1 = random.uniform(-1, 1)
# w2 = random.uniform(-1, 1)
# bias = random.uniform(-1, 1)
#
# # ---------------------------------------------------------------
# # 4. Training data: the four AND examples.
# # ---------------------------------------------------------------
# training_data = [
#     (0, 0, 0),
#     (0, 1, 0),
#     (1, 0, 0),
#     (1, 1, 1),
# ]
#
# learning_rate = 0.5  # how big a nudge we make each time
# epochs = 500000  # how many times we loop over the data
#
# # ---------------------------------------------------------------
# # 5. The learning loop.
# # ---------------------------------------------------------------
# for epoch in range(epochs):
#     for x1, x2, target in training_data:
#         neuron = NeuronV1("N1", w1, w2, bias)
#         output = predict_v1(neuron, x1, x2)  # the guess
#         error = output - target  # how wrong we were
#
#         # delta = error scaled by how much a weight change actually
#         # moves the output right here (the sigmoid slope).
#         delta = error * sigmoid_slope(output)
#
#         # Nudge each weight in the direction that reduces error.
#         # A weight only matters as much as its input was "on".
#         w1 -= learning_rate * delta * x1
#         w2 -= learning_rate * delta * x2
#         bias -= learning_rate * delta * 1  # bias input is always 1
#
#     # Every so often, print how the neuron is doing.
#     if epoch % 2000 == 0:
#         neuron = NeuronV1("N1", w1, w2, bias)
#         total_error = sum(
#             (predict_v1(neuron, a, b) - t) ** 2 for a, b, t in training_data
#         )
#         print(f"epoch {epoch:5d}   error {total_error:.8f}")
#
# # ---------------------------------------------------------------
# # 6. See what it learned.
# # ---------------------------------------------------------------
# print("\nFinal learned weights:")
# print(f"  w1 = {w1:.3f}   w2 = {w2:.3f}   bias = {bias:.3f}\n")
#
# neuron = NeuronV1("N1", w1, w2, bias)
# save_neuron(neuron)
#
# print("Testing the trained neuron:")
# for x1, x2, target in training_data:
#     out = predict_v1(neuron, x1, x2)
#     print(f"  {x1} AND {x2}  ->  {out:.3f}  (rounds to {round(out)}, want {target})")


if __name__ == '__main__':
    neuron_trainer = NeuronTrainer("N5")
    neuron_trainer.start_training(100)
    #                               42074000
    neuron_trainer.save()

    training_data_2_path = Path(__file__).resolve().parent / "src" / "train" / "training_data_2.json"
    neuron_trainer2 = NeuronTrainer("N4", training_data_2_path)
    neuron_trainer2.start_training(1)
    neuron_trainer2.save()