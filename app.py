from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

weights = np.load("mnist_weights.npz")

W1 = weights["W1"]
b1 = weights["b1"]
W2 = weights["W2"]
b2 = weights["b2"]
W3 = weights["W3"]
b3 = weights["b3"]


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def run_prediction(img_array):
    z1 = np.dot(img_array, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    output = softmax(z3[0])

    predicted_digit = int(np.argmax(output))
    confidence = float(np.max(output))

    return predicted_digit, confidence


@app.route("/")
def home():
    return "MNIST API is running!"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    image = Image.open(file).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(image).astype("float32") / 255.0

    # MNIST style: bright digit on dark background
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    img_array[img_array < 0.15] = 0.0
    img_array = img_array.reshape(1, 784)

    digit, confidence = run_prediction(img_array)

    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)