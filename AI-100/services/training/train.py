import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from services.prediction import predict_z, sigmoid

learning_rate = 0.5

def error_calculation(target, x1, x2):
    return predict_z(x1,x2) - target

def training():

    target = 1
    x1 = 1
    x2 = 0

    error = error_calculation(target, x1, x2)
    z = predict_z(x1, x2)

    sigomid_slope = z * (1 - z)

    delta = error * sigomid_slope

    w1 = learning_rate * delta * x1
    w2 = learning_rate * delta * x2
    bias = learning_rate * delta