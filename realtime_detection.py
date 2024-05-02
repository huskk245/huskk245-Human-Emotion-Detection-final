import cv2
from keras.models import model_from_json
import numpy as np

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


# Open the webcam
webcam = cv2.VideoCapture(0)

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

# Loop to capture frames from webcam
while True:
    # Read frame from webcam
    ret, im = webcam.read()
    if not ret:
        break  # Break if frame not captured

    # Convert frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    # Process each detected face
    for p, q, r, s in faces:
        # Extract the face region
        face_image = gray[q : q + s, p : p + r]

        # Draw rectangle around the face (changed color to green)
        cv2.rectangle(im, (p, q), (p + r, q + s), (0, 255, 0), 2)

        # Resize the face image to match model input size
        face_image = cv2.resize(face_image, (48, 48))

        # Extract features and normalize
        img = extract_features(face_image)

        # Make prediction
        pred = model.predict(img)

        # Get the predicted label
        prediction_label = labels[pred.argmax()]

        # Put predicted label on the frame
        cv2.putText(
            im,
            "% s" % (prediction_label),
            (p - 10, q - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            2,
            (0, 0, 255),
        )

    # Show the frame with predictions
    cv2.imshow("Output", im)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
