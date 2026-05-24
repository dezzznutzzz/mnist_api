from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Allow Flutter Web / browser requests
CORS(app, resources={r"/*": {"origins": "*"}})

model = load_model("mnist_model.keras")


@app.route("/")
def home():
    return "MNIST API is running!"


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handles browser preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    image = Image.open(file).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(image).astype("float32") / 255.0

    # MNIST style: white digit on black background
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array

    img_array[img_array < 0.15] = 0.0
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array, verbose=0)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "prediction": predicted_digit,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)