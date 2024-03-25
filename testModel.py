import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Load pre-trained face recognition model
# Replace 'path_to_model' with the actual path to your trained model
model = tf.keras.models.load_model('./face_recognition_model.h5')

# Dictionary to map predicted labels to names
label_to_name = {0: 'Waibhav', 1: 'Ashish', 2: 'Eshaan', 3: 'Aditya'}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess input image for the model
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (96, 96))
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict face using the model
def predict_face(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    label = np.argmax(prediction)
    return label_to_name[label]

# Open camera
cap = cv2.VideoCapture(0)

# Function to display the video stream with predictions
def display_video_stream():
    ret, frame = cap.read()
    if not ret:
        return None

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw green frame around faces and predict
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = frame[y:y + h, x:x + w]
        predicted_name = predict_face(face_image)
        cv2.putText(frame, f"This is {predicted_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Display video stream with Matplotlib
while True:
    frame = display_video_stream()
    if frame is None:
        break

    # Convert frame to RGB for Matplotlib
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display frame using Matplotlib
    plt.imshow(frame)
    plt.axis('off')
    plt.show()

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
