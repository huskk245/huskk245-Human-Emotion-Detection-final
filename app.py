import cv2
import numpy as np
from keras.models import model_from_json
from flask import Flask, request, jsonify, render_template, Response
from tempfile import NamedTemporaryFile

app = Flask(__name__)


class EmotionDetector(object):
    def __init__(self):
        # Load the model from JSON file
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("emotiondetector.h5")

        # Load the face cascade classifier
        haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_file)

        # Define labels for emotions
        self.labels = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise",
        }

    # Function to extract features from image
    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    # Function to make prediction on an image
    def predict_emotion(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)

        emotions = []
        # Process each detected face
        for p, q, r, s in faces:
            # Extract the face region
            face_image = gray[q : q + s, p : p + r]

            # Resize the face image to match model input size
            face_image = cv2.resize(face_image, (48, 48))

            # Extract features and normalize
            img = self.extract_features(face_image)

            # Make prediction
            pred = self.model.predict(img)

            # Get the predicted label
            prediction_label = self.labels[pred.argmax()]
            emotions.append(prediction_label)

            # Draw rectangle around the face
            cv2.rectangle(image, (p, q), (p + r, q + s), (0, 255, 0), 2)

            # Put predicted label on the image
            cv2.putText(
                image,
                "%s" % (prediction_label),
                (p - 10, q - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 255),
            )

        return image, list(set(emotions))

    def detect_emotion(self):
        # Open the webcam
        webcam = cv2.VideoCapture(0)

        # Loop to capture frames from webcam
        while True:
            # Read frame from webcam
            ret, im = webcam.read()
            if not ret:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Process each detected face
            for p, q, r, s in faces:
                # Extract the face region
                face_image = gray[q : q + s, p : p + r]

                # Resize the face image to match model input size
                face_image = cv2.resize(face_image, (48, 48))

                # Extract features and normalize
                img = self.extract_features(face_image)

                # Make prediction
                pred = self.model.predict(img)

                # Get the predicted label
                prediction_label = self.labels[pred.argmax()]

                # Put predicted label on the frame
                cv2.putText(
                    im,
                    "% s" % (prediction_label),
                    (p - 10, q - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    2,
                    (0, 0, 255),
                )

                # Draw rectangle around the face
                cv2.rectangle(im, (p, q), (p + r, q + s), (0, 255, 0), 2)

            # Convert frame to JPEG
            ret, jpeg = cv2.imencode(".jpg", im)

            # Yield the frame as byte array
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

    def get_frame(self):
        return self.detect_emotion()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/index2.html")
def index2():
    return render_template("index2.html")


@app.route("/index3.html")
def index3():
    return render_template("index3.html")


def gen(detector):
    return detector.get_frame()


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(EmotionDetector()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    if "image" not in request.files:
        return jsonify({"error": "No image found"}), 400

    image_file = request.files["image"]
    try:
        detector = EmotionDetector()
        # Read the image
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Make prediction on the image
        output_image, emotions = detector.predict_emotion(image)
        print(emotions)

        # Save the output image temporarily
        temp_image = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_path = temp_image.name
        cv2.imwrite(temp_path, output_image)

        return jsonify({"emotions": emotions, "image_path": temp_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
