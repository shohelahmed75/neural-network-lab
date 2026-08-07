"""
A small NEURAL NETWORK built from scratch that solves XOR.
No libraries doing the thinking — plain Python + math.exp.

Shape:  2 inputs  ->  2 hidden neurons  ->  1 output neuron

    x1 --\\        /--> [h1] --\
          >------<             >--> [o] --> output
    x2 --/        \\--> [h2] --/

A single neuron couldn't solve XOR (no straight line separates it).
Two layers CAN bend the boundary. The trick to training them is
BACKPROPAGATION: compute the error at the output, then push it
BACKWARD through the layers so each weight learns its share of blame.

Every weight below has its own name on purpose, so you can trace
exactly what happens. This is the same math as the single neuron,
just applied layer by layer.
"""

import math
import random


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_slope(output):
    # slope of sigmoid at this point, written using the output itself
    return output * (1 - output)


# ---------------------------------------------------------------
# The network's brain: every weight + bias, all starting random.
# ---------------------------------------------------------------
# Hidden neuron 1
w_h1_x1 = random.uniform(-1, 1)
w_h1_x2 = random.uniform(-1, 1)
b_h1    = random.uniform(-1, 1)
# Hidden neuron 2
w_h2_x1 = random.uniform(-1, 1)
w_h2_x2 = random.uniform(-1, 1)
b_h2    = random.uniform(-1, 1)
# Output neuron (reads the two hidden outputs)
w_o_h1  = random.uniform(-1, 1)
w_o_h2  = random.uniform(-1, 1)
b_o     = random.uniform(-1, 1)


def forward(x1, x2):
    # ---- FORWARD PASS: inputs flow to the output ----
    h1 = sigmoid(x1 * w_h1_x1 + x2 * w_h1_x2 + b_h1)
    h2 = sigmoid(x1 * w_h2_x1 + x2 * w_h2_x2 + b_h2)
    o  = sigmoid(h1 * w_o_h1  + h2 * w_o_h2  + b_o)
    return h1, h2, o


# XOR: output 1 only when the inputs DIFFER.
training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

learning_rate = 0.5
epochs = 60000


for epoch in range(epochs):
    for x1, x2, target in training_data:
        # ---- FORWARD ----
        h1, h2, o = forward(x1, x2)

        # ---- BACKWARD: start at the output ----
        error = o - target                       # how wrong the output was
        delta_o = error * sigmoid_slope(o)        # blame at the output neuron

        # Push the blame BACK to each hidden neuron.
        # A hidden neuron is blamed in proportion to the weight that
        # carried its signal forward to the output.
        delta_h1 = (delta_o * w_o_h1) * sigmoid_slope(h1)
        delta_h2 = (delta_o * w_o_h2) * sigmoid_slope(h2)

        # ---- UPDATE every weight (grad = delta * the input it multiplied) ----
        # Output neuron's weights
        w_o_h1 -= learning_rate * delta_o * h1
        w_o_h2 -= learning_rate * delta_o * h2
        b_o    -= learning_rate * delta_o * 1

        # Hidden neuron 1's weights
        w_h1_x1 -= learning_rate * delta_h1 * x1
        w_h1_x2 -= learning_rate * delta_h1 * x2
        b_h1    -= learning_rate * delta_h1 * 1

        # Hidden neuron 2's weights
        w_h2_x1 -= learning_rate * delta_h2 * x1
        w_h2_x2 -= learning_rate * delta_h2 * x2
        b_h2    -= learning_rate * delta_h2 * 1

    if epoch % 5000 == 0:
        total_error = sum((forward(a, b)[2] - t) ** 2 for a, b, t in training_data)
        print(f"epoch {epoch:6d}   error {total_error:.4f}")


print("\nTesting the trained network on XOR:")
for x1, x2, target in training_data:
    o = forward(x1, x2)[2]
    print(f"  {x1} XOR {x2}  ->  {o:.3f}  (rounds to {round(o)}, want {target})")

print(
    "\nIf the error got stuck (e.g. hovering near 1.0) instead of "
    "shrinking toward 0, just run it again — a bad random start can\n"
    "trap it in a dead end. That randomness is itself a real lesson."
)