import tensorflow as tf
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_bootstrap import Bootstrap
from tensorflow.keras.preprocessing import image

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class_names = ["Angry", "Fear", "Happy", "Sad", "Suprise"]


app = Flask(__name__)
bootstrap = Bootstrap(app)


model = tf.keras.models.load_model("models/emotion_model.keras")

IMG_SIZE = (128, 128)  # match training


def preprocess_uploaded_image(file):
    img = tf.keras.utils.load_img(file, color_mode="grayscale", target_size=IMG_SIZE)

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.cast(img_array, tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)

    return img_array


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "file not provided"}), 400

    file = request.files["file"]

    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file format"}), 400

    img = preprocess_uploaded_image(file)
    probs = model.predict(img, verbose=0)[0]

    label_index = int(tf.argmax(probs))
    label = class_names[label_index]
    confidence = float(probs[label_index])

    return jsonify({"class": label, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True)
