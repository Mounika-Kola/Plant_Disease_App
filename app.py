from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -----------------------------
# Load TFLite model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load Class Labels
# -----------------------------
with open(os.path.join(os.path.dirname(__file__), "class_labels.json"), "r") as f:
    class_names = json.load(f)

# -----------------------------
# Pesticide Mapping Dictionary
# -----------------------------
pesticide_map = {
    "Apple___Apple_scab": "Apply Mancozeb spray",
    "Apple___Black_rot": "Use Captan fungicide",
    "Apple___Cedar_apple_rust": "Use Myclobutanil fungicide",
    "Apple___healthy": "No pesticide needed",
    "Blueberry___healthy": "No pesticide needed",
    "Cherry_(including_sour)___Powdery_mildew": "Apply Sulfur fungicide",
    "Cherry_(including_sour)___healthy": "No pesticide needed",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply Azoxystrobin",
    "Corn_(maize)___Common_rust_": "Apply Mancozeb",
    "Corn_(maize)___Northern_Leaf_Blight": "Use Propiconazole",
    "Corn_(maize)___healthy": "No pesticide needed",
    "Grape___Black_rot": "Use Captan fungicide",
    "Grape___Esca_(Black_Measles)": "Apply systemic fungicide (e.g., Triazoles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply Copper fungicide",
    "Grape___healthy": "No pesticide needed",
    "Orange___Haunglongbing_(Citrus_greening)": "No chemical control, use resistant varieties",
    "Peach___Bacterial_spot": "Apply Copper-based bactericide",
    "Peach___healthy": "No pesticide needed",
    "Pepper,_bell___Bacterial_spot": "Apply Copper fungicide",
    "Pepper,_bell___healthy": "No pesticide needed",
    "Potato___Early_blight": "Apply Mancozeb",
    "Potato___Late_blight": "Use Chlorothalonil",
    "Potato___healthy": "No pesticide needed",
    "Raspberry___healthy": "No pesticide needed",
    "Soybean___healthy": "No pesticide needed",
    "Squash___Powdery_mildew": "Apply Sulfur fungicide",
    "Strawberry___Leaf_scorch": "Apply Captan fungicide",
    "Strawberry___healthy": "No pesticide needed",
    "Tomato___Bacterial_spot": "Apply Copper fungicide",
    "Tomato___Early_blight": "Apply Mancozeb",
    "Tomato___Late_blight": "Use Chlorothalonil",
    "Tomato___Leaf_Mold": "Apply Copper fungicide",
    "Tomato___Septoria_leaf_spot": "Use Chlorothalonil spray",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply Abamectin",
    "Tomato___Target_Spot": "Use Mancozeb",
    "Tomato___Tomato_mosaic_virus": "No chemical control, use resistant varieties",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No chemical control, use resistant varieties",
    "Tomato___healthy": "No pesticide needed",
}

# -----------------------------
# Helper Functions
# -----------------------------
def predict(img):
    # Preprocess image
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]
    class_idx = np.argmax(prediction)
    confidence = round(100 * np.max(prediction), 2)

    return class_names[class_idx], confidence

def recommend_pesticide(disease_class):
    return pesticide_map.get(disease_class, "No recommendation available")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files['file']
    img = Image.open(file).resize((256, 256)).convert("RGB")

    predicted_class, confidence = predict(img)
    pesticide = recommend_pesticide(predicted_class)

    return jsonify({
        "disease": predicted_class,
        "pesticide": pesticide,
        "confidence": f"{confidence}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
