import cv2
from keras.models import model_from_json
import numpy as np
from tkinter import Tk, filedialog

# Load the model from JSON file
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the face cascade classifier
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)


# Function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# Function to make prediction on an image
def predict_emotion(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    # Process each detected face
    for p, q, r, s in faces:
        # Extract the face region
        face_image = gray[q : q + s, p : p + r]

        # Resize the face image to match model input size
        face_image = cv2.resize(face_image, (48, 48))

        # Extract features and normalize
        img = extract_features(face_image)

        # Make prediction
        pred = model.predict(img)

        # Get the predicted label
        prediction_label = labels[pred.argmax()]

        # Draw rectangle around the face
        cv2.rectangle(image, (p, q), (p + r, q + s), (0, 255, 0), 2)

        # Put predicted label on the image
        cv2.putText(
            image,
            "% s" % (prediction_label),
            (p - 10, q - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 0, 255),
        )

    return image


# Define labels for emotions
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Create Tkinter window for file dialog
root = Tk()
root.withdraw()  # Hide the main window

# Open file dialog to select image
file_path = filedialog.askopenfilename()

# Read the image
image = cv2.imread(file_path)

# Make prediction on the image
output_image = predict_emotion(image)

# Display the output image
cv2.imshow("Output", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
