# cnn_training.py

import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# Parameters
image_size = 128
batch_size = 32
epochs = 15

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage\\train",
    image_size=(image_size, image_size),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage\\val",
    image_size=(image_size, image_size),
    batch_size=batch_size
)

# Normalize + Augmentation
resize_rescale = tf.keras.Sequential([
    layers.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2)
])

# CNN Model
cnn_model = models.Sequential([
    resize_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_ds.class_names), activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train
history = cnn_model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs)

# Save model
cnn_model.save("cnn_plant_disease_model.h5")

# Accuracy Graph
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Loss')
plt.legend()
plt.show()
