from keras.models import load_model
import numpy as np

model = load_model("mnist_model.keras", compile=False)

weights = model.get_weights()

np.savez(
    "mnist_weights.npz",
    W1=weights[0],
    b1=weights[1],
    W2=weights[2],
    b2=weights[3],
    W3=weights[4],
    b3=weights[5],
)

print("Weights exported successfully!")