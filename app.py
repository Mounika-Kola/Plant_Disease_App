from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)

# -----------------------------
# Load Class Labels
# -----------------------------
with open("class_labels.json", "r") as f:
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
# Helper Function
# -----------------------------
def recommend_pesticide(disease_class):
    return pesticide_map.get(disease_class, "No recommendation available")

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = Image.open(file).resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    disease = class_names[class_idx]
    pesticide = recommend_pesticide(disease)
    confidence = f"{np.max(prediction)*100:.2f}%"

    return jsonify({
        "disease": disease,
        "pesticide": pesticide,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)