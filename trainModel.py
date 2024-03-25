import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load and preprocess images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (100, 100))  # Resize to a fixed size
        if img is not None:
            images.append(img)
    return images

# Load images from folders
waibhav_images = load_images_from_folder('./images/Waibhav')
ashish_images = load_images_from_folder('./images/Ashish')
eshaan_images = load_images_from_folder('./images/Eshaan')
aditya_images = load_images_from_folder('./images/Aditya')

# Create labels for each person
waibhav_labels = np.zeros(len(waibhav_images))
ashish_labels = np.ones(len(ashish_images))
eshaan_labels = np.full(len(eshaan_images), 2)
aditya_labels = np.full(len(aditya_images), 3)

# Combine images and labels
X = np.concatenate([waibhav_images, ashish_images, eshaan_images, aditya_images])
y = np.concatenate([waibhav_labels, ashish_labels, eshaan_labels, aditya_labels])

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert images and labels to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# Model architecture
model = models.Sequential([
    layers.Input(shape=(100, 100, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 output classes
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Preprocess and batch the datasets
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float
    return image, label

train_dataset = train_dataset.map(preprocess_image).batch(32)
val_dataset = val_dataset.map(preprocess_image).batch(32)

# Train model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Save model
model.save('face_recognition_model.h5')
