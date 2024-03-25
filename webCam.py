import cv2
import os
import matplotlib.pyplot as plt

# Function to create a folder to store images
def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)

# Function to capture images of the person
def capture_images(name):
    # Create a folder for the person
    folder_path = os.path.join("images", name)
    create_folder(folder_path)

    # Initialize camera
    camera = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0
    while count < 500:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Save images without the green rectangle
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(folder_path, f"face_{count}.jpg"), face_img)
            count += 1

        # Display frame in the "Capturing" window
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Capturing')
        plt.pause(0.01)

    # Release camera
    camera.release()

def main():
    # Take the name of the person as input
    name = input("Enter the name of the person: ")

    # Capture images of the person
    capture_images(name)

if __name__ == "__main__":
    main()
