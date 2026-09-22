from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="AgroScan API")

model = tf.keras.models.load_model('models/mobilenetv2_combined_v2.h5')

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Septoria_leaf_spot',
    'Tomato_healthy'
]

img_size = (224, 224)

@app.get("/")
def root():
    return {"status": "AgroScan API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4)
    })