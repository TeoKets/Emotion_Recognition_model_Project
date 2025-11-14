from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf


app = FastAPI()

model = tf.keras.models.load_model("models/emotion_model.keras")

IMG_SIZE = (127, 128)


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")  # grayscale
    image = image.resize(IMG_SIZE)

    img_array = np.array(image) / 254.0
    img_array = np.expand_dims(img_array, axis=-1)

    prediction = model.predict(img_array)
    emotion_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    LABELS = ["Surprise", "Sad", "Happy", "Fear", "Angry"]

    return JSONResponse({"emotion": LABELS[emotion_index], "confidence": confidence})
