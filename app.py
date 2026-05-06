from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

def predict(disease_model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = disease_model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

app = Flask(__name__)

# -----------------------------
# Load Models
# -----------------------------
leaf_model = tf.keras.models.load_model("leaf_detector.h5")
disease_model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)

# -----------------------------
# Load Class Labels
# -----------------------------
with open("class_labels.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# Pesticide Mapping Dictionary
# -----------------------------
pesticide_map = {
    "Apple___Apple_scab": [
        "Copper hydroxide (applied at green tip stage as a protective organic fungicide)",
        "Trichoderma harzianum (biological agent applied at petal fall for protection)"
    ],
    "Apple___Black_rot": [
        "HarzShield (Trichoderma harzianum foliar spray and soil drench)",
        "Bacillus subtilis (Serenade Garden AgraQuest) foliar application"
    ],
    "Apple___Cedar_apple_rust": [
        "Bonide Captain Jack’s Copper Fungicide (copper-based, OMRI-listed, applied at pre-bloom and post-bloom)",
        "Sulfur-based organic fungicide (applied at early leaf stage and repeated as needed)"
    ],
    "Apple___healthy": [
        "No Treatment Needed"
    ],
    "Blueberry___healthy": [
        "No Treatment Needed"
    ],
    "Cherry_(including_sour)___Powdery_mildew": [
        "Sulfur-based organic fungicide spray (applied at first sign of disease and repeated every 7–10 days)",
        "Bacillus subtilis (Serenade ASO or similar) foliar spray"
    ],
    "Cherry_(including_sour)___healthy": [
        "No Treatment Needed"
    ],
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        "Copper hydroxide foliar spray (applied at early disease onset, OMRI-listed)",
        "Trichoderma spp. (foliar and soil application as a biocontrol agent)"
    ],
    "Corn_(maize)___Common_rust_": [
        "Trichoderma spp. foliar spray (biocontrol agent targeting rust pathogens)",
        "Pseudomonas fluorescens seed treatment (to suppress seed- and soil-borne inoculum)"
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        "Trichoderma asperellum 576 conidial suspension (10^7 spores/mL, foliar spray and soil drench)",
        "Pseudomonas fluorescens foliar spray (antagonistic bacteria applied at early symptom appearance)"
    ],
    "Corn_(maize)___healthy": [
        "No Treatment Needed"
    ],
    "Grape___Black_rot": [
        "Trichoderma asperellum (MCBY2) bioformulation (applied as foliar spray)",
        "Bacillus subtilis (SB2) bioformulation (applied as foliar spray)"
    ],
    "Grape___Esca_(Black_Measles)": [
        "Bio-Tam 2.0 (Trichoderma asperellum and T. gamsii) applied post-pruning",
        "Bacillus subtilis (SB5) bioformulation (applied as foliar spray)"
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        "Trichoderma viride (DRRS1) bioformulation (foliar application)",
        "Bacillus licheniformis (RB1) bioformulation (foliar application)"
    ],
    "Grape___healthy": [
        "No Treatment Needed"
    ],
    "Orange___Haunglongbing_(Citrus_greening)": [
        "Beauveria bassiana (strain 2067) conidial suspension (1 × 10^7 conidia/mL, foliar spray for vector control)",
        "Neem oil spray (acts as a repellent and disrupts psyllid feeding and reproduction)"
    ],
    "Peach___Bacterial_spot": [
        "Cellulose nanofiber spray (85% control efficacy against Xanthomonas arboricola pv. pruni)",
        "Bacillus amyloliquefaciens / Bacillus subtilis-based biocontrol agents (foliar application)"
    ],
    "Peach___healthy": [
        "No Treatment Needed"
    ],
    "Pepper,_bell___Bacterial_spot": [
        "Soil drenching with Bacillus subtilis B01 (biocontrol agent)",
        "Foliar spray with 0.5 mM salicylic acid (plant defense inducer)"
    ],
    "Pepper,_bell___healthy": [
        "No Treatment Needed"
    ],
    "Potato___Early_blight": [
        "Seed treatment with Trichoderma viride at 0.5% concentration (before planting)",
        "Foliar spray of Neem Seed Kernel Extract (NSKE) at 5% (applied at early disease onset)"
    ],
    "Potato___Late_blight": [
        "ChiProPlant (Chitosan hydrochloride, applied as foliar spray at label rates)",
        "TC 4 (Trichoderma atroviride, applied as foliar spray and soil drench)"
    ],
    "Potato___healthy": [
        "No Treatment Needed"
    ],
    "Raspberry___healthy": [
        "No Treatment Needed"
    ],
    "Soybean___healthy": [
        "No Treatment Needed"
    ],
    "Squash___Powdery_mildew": [
        "Kaligreen (82% potassium bicarbonate, foliar spray at label rates)",
        "Mildew Cure (30% cottonseed oil, 30% corn oil, 23% garlic extract, foliar spray)"
    ],
    "Strawberry___Leaf_scorch": [
        "Serifel (9.9% Bacillus amyloliquefaciens strain MBI 600, foliar spray)",
        "Serenade ASO (1.34% Bacillus subtilis strain QST 713, foliar spray)"
    ],
    "Strawberry___healthy": [
        "No Treatment Needed"
    ],
    "Tomato___Bacterial_spot": [
        "GreenFurrow BacStop (clove, rosemary, peppermint, cottonseed, thyme, garlic, cinnamon oils blend, foliar spray)",
        "Brandt Organics Aleo (78% garlic oil, foliar spray)"
    ],
    "Tomato___Early_blight": [
        "Promax (3.5% thyme oil, foliar spray at label rates)",
        "Regalia (5% extract of Reynoutria sachalinensis, foliar spray)"
    ],
    "Tomato___Late_blight": [
        "RootShield Plus WP (1.15% Trichoderma harzianum Rifai strain T-22 and 0.61% Trichoderma virens strain G-41, soil and foliar application)",
        "TerraClean 5.0 (27% hydrogen dioxide and 5% peroxyacetic acid, foliar spray)"
    ],
    "Tomato___Leaf_Mold": [
        "Serenade Opti (26.2% Bacillus subtilis strain QST 713, foliar spray)",
        "OxiDate 2.0 (27% hydrogen dioxide and 2% peroxyacetic acid, foliar spray)"
    ],
    "Tomato___Septoria_leaf_spot": [
        "Double Nickel 55 (Bacillus amyloliquefaciens strain D747, foliar spray)",
        "Procidic (3.5% citric acid, foliar spray)"
    ],
    "Tomato___Spider_mites Two-spotted_spider_mite": [
        "ECOWORKS EC (70% cold pressed neem oil, foliar spray)",
        "Organic JMS Stylet-oil (97.1% paraffinic oil, foliar spray)"
    ],
    "Tomato___Target_Spot": [
        "Trilogy (70% clarified hydrophobic extract of neem oil, foliar spray)",
        "Sporan EC2 (16% rosemary oil, 10% clove oil, 10% thyme oil, 2% peppermint oil, foliar spray)"
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        "Thyme Guard (23% thyme oil extract, foliar spray to reduce vector transmission)",
        "Organic JMS Stylet-oil (97.1% paraffinic oil, foliar spray to deter whiteflies)"
    ],
    "Tomato___Tomato_mosaic_virus": [
        "Thyme Guard (23% thyme oil extract, foliar spray to reduce virus spread)",
        "Organic JMS Stylet-oil (97.1% paraffinic oil, foliar spray to reduce mechanical transmission)"
    ],
    "Tomato___healthy": [
        "No Treatment Needed"
    ]
}

# -----------------------------
# Helper Functions
# -----------------------------
def recommend_pesticide(disease_class):
    return pesticide_map.get(disease_class, "No recommendation available")

def preprocess_image(file, target_size=(224,224)):
    img = Image.open(file).convert("RGB")
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)  # batch dimension
    return img_array

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
def predict_route():
    file = request.files['file']

    # Step 1: Leaf check (224x224)
    leaf_array = preprocess_image(file, target_size=(224,224))
    leaf_pred = leaf_model.predict(leaf_array)
    print("Leaf prediction raw:", leaf_pred)  # Debug

    # If model outputs a single sigmoid probability
    if leaf_pred.shape[1] == 1:
        leaf_prob = leaf_pred[0][0]
        if leaf_prob >= 0.5:  # Not a leaf
            return jsonify({
                "error": "No leaf detected, please upload or capture a leaf image."
            })

    # Step 2: Disease prediction (224x224)
    img = Image.open(file).resize((256, 256))   # match training size!
    img = img.convert("RGB")                    # ensure 3 channels

    predicted_class, confidence = predict(disease_model, img)

    pesticide = recommend_pesticide(predicted_class)

    return jsonify({
        "disease": predicted_class,
        "pesticide": pesticide,
        "confidence": f"{confidence}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
