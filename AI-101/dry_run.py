from src.neuron_loader import load_neuron_model
from src.services.prediction_service import predict_v1

if __name__ == '__main__':
    neuron = load_neuron_model("N5")
    if not neuron:
        print("Neuron model not loaded")
        exit(0)

    x1 = int(input("Enter first value: "))
    x2 = int(input("Enter second value: "))

    out = predict_v1(neuron, x1, x2)
    print(f"  {x1} AND {x2}  ->  {out:.10f}  (rounds to {round(out)})")