import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

# Build class_labels automatically from dataset folders
train_dir = "PlantVillage/train"
class_labels = sorted(os.listdir(train_dir))

# Load and preprocess test image (resize to 128x128, same as training)
img_path = r"test sample.jpg"
img = image.load_img(img_path, target_size=(128, 128))   # <-- changed to 128
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

print("Predicted class index:", predicted_class)
print("Predicted class label:", class_labels[predicted_class])